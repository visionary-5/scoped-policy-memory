"""CAP-CPT reproduction (Category-Aware Policy Continued Pre-Training, arXiv 2510.11588) on CC-Gen-ST-Lite.

Pipeline (paper Sec. 3.2, Fig. 2, App. C/D/E) and this implementation:
 1. Policy analysis: an LLM splits the policy into facts / behaviours / simple / complex conditions using
    the official prompt (``Utils/Policy_Analysis/Final_Prompting.md`` of the CAP-CPT release; pass it
    with ``--prompt-file``). The LLM analysis is only reported descriptively (agreement with the
    structural ground truth); the CPT data is built from the structural ground truth (= the paper's
    optional manual check, which favours the baseline).
 2. Category-specific CPT data, with the policy always referred to by its id ``#Pxxxxx``:
    paraphrases (all types, LLM) + QA (all types, both branches of conditions; question rewrites by the
    LLM, answers computed by program); behaviour demonstrations (1K per behavioural spec); scenario
    simulations (5K per condition spec: LLM-written question templates, train-split profiles,
    program-computed answer with a one-line derivation); trajectory familiarisation (= the SFT rows).
    ``nosim`` drops the scenario simulations (paper ablation).
 3. CPT (all tokens, 1 pass) -> SFT (800 steps, same as the other arms) continuing the same LoRA.
    Inference uses the anchor sentence "Based on the policy document #P... you previously learnt about".

Arms: ``sftA`` (SFT only on P_anchor), ``capcpt`` (CPT full -> SFT P_anchor), ``capcpt_nosim``
(CPT nosim -> SFT P_anchor), ``capcpt_conf`` (CPT full -> SFT prec_anchor = P_anchor + the 128
conflict rows of prec verbatim; tests whether our conflict samples stack on CAP-CPT).
"""
from __future__ import annotations

import collections
import json
import random
import re
from pathlib import Path

from ..data.ccgen import (LAYERS, _names, env_of, op_text, policy_block, render_profile, request_templates,
                          sample_profile, solve, tasks)
from ..data.prompts import build_messages
from ..eval.score import render_calls

N_PARA, N_QREW = 5, 3
N_SCENARIO_PER_SPEC, N_BEHAVIOR_PER_SPEC = 5000, 1000
ARM_SPEC = {'sftA': (None, 'P_anchor'), 'capcpt': ('cpt_full', 'P_anchor'),
            'capcpt_conf': ('cpt_full', 'prec_anchor'), 'capcpt_nosim': ('cpt_nosim', 'P_anchor')}
CUT = 'Now you need to work with the following Policy Document:'


def pidstr(w):
    return f"#P{w.extra['policy_id']:05d}"


def anchor(w):
    return f"Based on the policy document {pidstr(w)} you previously learnt about, answer the user query."


# ---------------- spec inventory (structural ground truth) ----------------
def spec_list(w):
    g = next(r for r in w.rules if r['kind'] == 'ccgen_const'); T = w.extra['T']; out = []
    out.append({'sid': 'g_values', 'cat': 'fact', 'scope': 'all', 'text': g['text'].splitlines()[1], 'nums': [str(v) for v in g['values']], 'names': []})
    out.append({'sid': 'g_tasks', 'cat': 'fact', 'scope': 'all', 'text': g['text'].splitlines()[3], 'nums': [str(T)], 'names': ['Task_Type_1']})
    for t in tasks(w):
        k = t['task']; lines = [l for l in t['text'].splitlines() if not l.startswith('  - arg_')]
        out.append({'sid': f't{k}_access', 'cat': 'behavior', 'scope': f'Task_Type_{k}', 'task': k, 'layers': t['layers'], 'text': '\n'.join(lines),
                    'nums': [], 'names': []})
        for i, op in enumerate(t['ops'], 1):
            nums = [str(op['value'])] if op['type'] == 'constant' else [e for e in op.get('operands', []) if e.isdigit()]
            cat = 'cond_simple' if op['type'] == 'two_element' else 'cond_direct'
            out.append({'sid': f't{k}_a{i}', 'cat': cat, 'scope': f'Task_Type_{k}', 'task': k, 'arg': i, 'op': op,
                        'text': f"Task_Type_{k}, arg_{i} of finish_task_{k}: {op_text(op)}", 'nums': nums,
                        'names': [n for n in _names(op)] + [f'arg_{i}']})
    return out


def qa_list(w, specs):
    """Templated QA with program answers; conditional specs cover both branches."""
    P = pidstr(w); g = next(r for r in w.rules if r['kind'] == 'ccgen_const'); T = w.extra['T']; qa = []
    for i, v in enumerate(g['values'], 1):
        qa.append(('g_values', f"What is the value of global_attribute_{i} in Policy {P}?", f"In Policy {P}, global_attribute_{i} = {v}."))
    qa.append(('g_tasks', f"Which task types does Policy {P} cover?", f"Policy {P} lets the agent help users finish Task_Type_1 to Task_Type_{T}."))
    for s in specs:
        if s['cat'] == 'behavior':
            L = ', '.join(f'layer {l}' for l in s['layers']); k = s['task']
            qa.append((s['sid'], f"Which layers' profiles must be accessed for Task_Type_{k} under Policy {P}?",
                       f"For Task_Type_{k}, Policy {P} requires one profile instance from each of {L}, and the agent calls finish_task_{k} with arguments from one instance per layer at a time."))
        elif s['cat'].startswith('cond'):
            k, i, op = s['task'], s['arg'], s['op']
            qa.append((s['sid'], f"How is arg_{i} of finish_task_{k} computed in Policy {P}?", f"In Policy {P}, arg_{i} of finish_task_{k} is: {op_text(op)}"))
            if op['type'] == 'two_element':
                a, b = op['operands']; big = op['op_type'] == 'max'
                qa.append((s['sid'], f"In Policy {P}, for arg_{i} of finish_task_{k}, which value is used when {a} is greater than {b}?", f"arg_{i} takes {a if big else b}."))
                qa.append((s['sid'], f"In Policy {P}, for arg_{i} of finish_task_{k}, which value is used when {a} is not greater than {b}?", f"arg_{i} takes {b if big else a}."))
    return qa


# ---------------- LLM prompts ----------------
CAT_NAME = {'fact': 'Fact Illustration', 'behavior': 'Behavior Specification', 'cond_simple': 'Workflow Specification (Simple)',
            'cond_direct': 'Workflow Specification (Simple)'}
FALLBACK_SCEN = ["Under {POLICY}, what value should be passed as {ARG} when calling the tool for {TASK}?",
                 "Given these selected profiles, what is {ARG} for {TASK} according to {POLICY}?",
                 "According to {POLICY}, compute {ARG} of the {TASK} tool call for this scenario.",
                 "What does {POLICY} say {ARG} should be for {TASK} in this case?"]
FALLBACK_BEH = ["I want to perform {TASK}.", "Please complete {TASK} with the selected profiles below.",
                "Do {TASK} for the profiles I selected.", "My profiles are selected; run {TASK}."]


def analysis_prompt(w, prompt_text):
    i = prompt_text.rfind(CUT)
    assert i > 0, 'official analysis prompt format changed'
    return prompt_text[:i + len(CUT)] + '\n\n' + policy_block(w.rules)


def prompts_for(w, specs, qa, prompt_text):
    P = pidstr(w)
    out = [{'pid': 'analysis', 'kind': 'analysis', 'prompt': analysis_prompt(w, prompt_text), 'max_tokens': 8192, 'temperature': 0.0}]
    for s in specs:
        out.append({'pid': 'para:' + s['sid'], 'kind': 'para', 'sid': s['sid'], 'max_tokens': 1024, 'temperature': 0.7, 'prompt':
            f"You are helping to create study material for an internal agent policy document, Policy {P}. Below is one specification from Policy {P}.\n\n"
            f"Specification type: {CAT_NAME[s['cat']]}. Valid scope: {s['scope']}.\n{s['text']}\n\n"
            f"Write {N_PARA} different paraphrases of this specification. Each paraphrase must keep exactly the same meaning, refer to the document as Policy {P}, "
            "and keep every attribute name (such as layer_1_attribute_7 or global_attribute_2), task name, argument name and number exactly as written. "
            f"Output only a numbered list from 1. to {N_PARA}., one paraphrase per line."})
    for j, (sid, q, _) in enumerate(qa):
        out.append({'pid': f'qrew:{j}', 'kind': 'qrew', 'sid': sid, 'q': q, 'max_tokens': 512, 'temperature': 0.7, 'prompt':
            f"Rewrite the following question about Policy {P} in {N_QREW} different ways without changing its meaning. Keep every attribute name, task name, "
            f"argument name and number exactly as written.\n\nQuestion: {q}\n\nOutput only a numbered list from 1. to {N_QREW}., one question per line."})
    out.append({'pid': 'scen_tmpl', 'kind': 'scen_tmpl', 'max_tokens': 1024, 'temperature': 0.7, 'prompt':
        "We are writing practice questions about an agent policy document. Each question comes right after a scenario that lists a user's selected profiles, "
        "and asks for the value of one argument of a task's tool call. Write 10 different natural ways to ask this question. Use the placeholders {POLICY}, {TASK} "
        "and {ARG} exactly once each ({POLICY} stands for a document name such as 'Policy #P12345', {TASK} for a task such as 'Task_Type_5', {ARG} for an argument "
        "such as 'arg_2'). Output only a numbered list from 1. to 10., one question per line."})
    out.append({'pid': 'beh_tmpl', 'kind': 'beh_tmpl', 'max_tokens': 1024, 'temperature': 0.7, 'prompt':
        "Write 10 different short ways a user could ask an agent to perform a task, where {TASK} is a placeholder for a task name such as task_type_5. "
        "The user's selected profiles will be listed right after the request, so do not include any profile values. Use the placeholder {TASK} exactly once. "
        "Output only a numbered list from 1. to 10., one request per line."})
    return out


def anchor_sets(w, P_rows, prec_rows):
    """P_anchor: P rows with the anchor sentence in place of any policy text (the paper's SFT format);
    prec_anchor: P_anchor + the 128 conflict rows of prec verbatim."""
    anc = anchor(w)
    conf = [r for r in prec_rows if r.get('source') in ('override', 'substitute')]
    assert len(conf) == 128 and collections.Counter(r['source'] for r in conf) == {'override': 64, 'substitute': 64}
    PA = []
    for r in P_rows:
        r = dict(r)
        if r.get('item'):
            r['messages'] = build_messages(r['item'], anc)
        PA.append(r)
    return {'P_anchor': PA, 'prec_anchor': sorted(PA + conf, key=lambda r: r['uid'])}


# ---------------- CPT corpus assembly ----------------
NUM = re.compile(r'^\s*(\d+)[\.\)]\s*(.+?)\s*$')


def numbered(text):
    return [m.group(2).strip().strip('"') for m in map(NUM.match, text.splitlines()) if m]


def has_num(s, n):
    return re.search(r'(?<![\w.])' + re.escape(n) + r'(?![\w])', s) is not None


def valid_para(s, p, P):
    if P not in p:
        return False
    if not all(has_num(p, n) for n in s['nums']) or not all(n in p for n in s['names']):
        return False
    if 'task' in s and not re.search(rf"(?:task_type_|finish_task_|task_|task type ){s['task']}(?!\d)", p, re.I):
        return False
    if s['cat'] == 'behavior':
        bad = [l for l in range(1, LAYERS + 1) if l not in s['layers'] and re.search(rf'layer[ _]?{l}\b', p)]
        if bad or not all(re.search(rf'\b{l}\b', p) for l in s['layers']):
            return False
    if s['cat'] == 'cond_simple' and s['op']['op_type'][:3] not in p.lower():
        return False
    return True


def keep_tokens(q):
    return re.findall(r'(?:layer_\d_attribute_\d|global_attribute_\d|arg_\d+|finish_task_\d+|Task_Type_\d+|#P\d{5}|(?<![\w.])\d+(?!\w))', q)


def norm(s):
    return re.sub(r'[^a-z0-9_ ]', '', s.lower()).strip()


def rationale(w, k, i, op, profile):
    env = env_of(w, profile); P = pidstr(w); head = f"Under Policy {P}, arg_{i} of finish_task_{k}"
    fv = lambda x: f'"{x}"' if isinstance(x, str) else str(x); res = solve(w, k, profile)[i - 1]; t = op['type']
    if t == 'value':
        s = f"{head} is the value of {op['attribute']}. Here {op['attribute']} = {fv(env[op['attribute']])}"
    elif t == 'lookup':
        s = f"{head} is the original lookup value of {op['attribute']} from the selected profile. Here {op['attribute']} = {fv(env[op['attribute']])}"
    elif t == 'constant':
        s = f"{head} is the constant {op['value']}"
    else:
        x, y = op['operands']; word = 'maximum' if op['op_type'] == 'max' else 'minimum'
        vals = ' and '.join(f"{e} = {env[e]}" for e in (x, y) if not e.isdigit())
        s = f"{head} is the {word} between {x} and {y}. Here {vals}"
    return f"{s}, so arg_{i} = {fv(res)}.", res


def assemble(w, specs, qa, raw: dict, test_rows, PA_rows, render_trajectory,
             n_scen=N_SCENARIO_PER_SPEC, n_beh=N_BEHAVIOR_PER_SPEC):
    """Build (docs_full, docs_nosim, report) for one seed. ``raw`` maps prompt id -> LLM output row
    (empty dict = built-in fallback templates only, for dry runs). ``render_trajectory(messages)``
    returns the chat-template text of an SFT prompt (tokenizer of the trained model)."""
    P = pidstr(w); seed = w.seed
    rep = {'n_scenario_per_spec': n_scen, 'n_behavior_per_spec': n_beh, 'fallback_only': not raw}
    rng = random.Random(f'capcpt-v1-cpt-{w.hash()}-{seed}'); docs = []
    kept = collections.Counter(); tried = collections.Counter()
    for s in specs:
        cand = numbered(raw.get('para:' + s['sid'], {}).get('output', ''))[:N_PARA]; tried[s['cat']] += len(cand)
        for p in dict.fromkeys(cand):
            if valid_para(s, p, P):
                docs.append({'src': 'paraphrase', 'sid': s['sid'], 'text': f"Policy {P} specification ({CAT_NAME[s['cat']]}), paraphrased: {p}"}); kept[s['cat']] += 1
        docs.append({'src': 'paraphrase', 'sid': s['sid'], 'text': f"Policy {P} specification ({CAT_NAME[s['cat']]}): {s['text']}"})
    rep['paraphrase'] = {'tried': dict(tried), 'kept': dict(kept)}
    nq = collections.Counter()
    for j, (sid, q, ans) in enumerate(qa):
        docs.append({'src': 'qa', 'sid': sid, 'text': f"Question: {q}\nAnswer: {ans}"}); nq['templated'] += 1
        need = keep_tokens(q)
        for q2 in dict.fromkeys(numbered(raw.get(f'qrew:{j}', {}).get('output', ''))[:N_QREW]):
            nq['tried'] += 1
            if q2 != q and all(t in q2 for t in need):
                docs.append({'src': 'qa', 'sid': sid, 'text': f"Question: {q2}\nAnswer: {ans}"}); nq['kept'] += 1
    rep['qa'] = dict(nq)
    st = [t for t in dict.fromkeys(numbered(raw.get('scen_tmpl', {}).get('output', ''))) if all(t.count(x) == 1 for x in ('{POLICY}', '{TASK}', '{ARG}')) and len(t) < 300]
    pools = request_templates(); held = {norm(x.format(k='{TASK}').replace('task_type_{TASK}', '{TASK}')) for sp in ('dev', 'test') for x in pools[sp]}
    bt = [t for t in dict.fromkeys(numbered(raw.get('beh_tmpl', {}).get('output', ''))) if t.count('{TASK}') == 1 and len(t) < 200 and norm(t) not in held]
    rep['templates'] = {'scen_valid': len(st), 'beh_valid': len(bt), 'scen_fallback': len(st) < 3, 'beh_fallback': len(bt) < 3}
    if len(st) < 3:
        st = st + FALLBACK_SCEN
    if len(bt) < 3:
        bt = bt + FALLBACK_BEH
    test_profiles = {render_profile(r['fields']['profile']) for r in test_rows}
    coll = 0
    for s in specs:  # behaviour demonstrations
        if s['cat'] != 'behavior':
            continue
        k = s['task']; L = ', '.join(f'layer {l}' for l in s['layers'])
        for _ in range(n_beh):
            pr = sample_profile(rng, 'train'); rp = render_profile(pr)
            if rp in test_profiles:
                coll += 1; continue
            req = rng.choice(bt).replace('{TASK}', f'task_type_{k}')
            ans = render_calls([[f'finish_task_{k}', {'attributes': solve(w, k, pr)}]])
            docs.append({'src': 'behavior', 'sid': s['sid'], 'text': f"Policy {P} behavior demonstration.\nUser: {req}\n{rp}\nAgent: Following Policy {P}, Task_Type_{k} needs one profile instance "
                         f"from each of {L}; the agent computes the arguments in the order the policy specifies and calls the tool for this profile combination: {ans}"})
    for r in PA_rows:  # trajectory familiarisation: the SFT rows as their exact training text
        docs.append({'src': 'trajectory', 'sid': r.get('rule_id'), 'text': render_trajectory(r['messages']) + r['oracle']})
    nosim = list(docs)
    for s in specs:  # scenario simulations
        if not s['cat'].startswith('cond'):
            continue
        k, i, op = s['task'], s['arg'], s['op']
        for _ in range(n_scen):
            pr = sample_profile(rng, 'train'); rp = render_profile(pr)
            if rp in test_profiles:
                coll += 1; continue
            q = rng.choice(st).replace('{POLICY}', f'Policy {P}').replace('{TASK}', f'Task_Type_{k}').replace('{ARG}', f'arg_{i}')
            why, res = rationale(w, k, i, op, pr); assert res == solve(w, k, pr)[i - 1]
            docs.append({'src': 'scenario', 'sid': s['sid'], 'text': f"Policy {P} scenario simulation.\nA user asks to perform task_type_{k}.\n{rp}\nQuestion: {q}\nAnswer: {why}"})
    rep['test_profile_collisions_dropped'] = coll
    out = {}
    for name, ds in (('full', docs), ('nosim', nosim)):
        ds = list(ds); random.Random(f'capcpt-v1-shuffle-{name}-{seed}').shuffle(ds)
        out[name] = ds
        rep[name] = {'n': len(ds), 'by_src': dict(collections.Counter(d['src'] for d in ds))}
    rep['analysis'] = analysis_agreement(raw.get('analysis', {}).get('output', ''), specs) if raw else None
    return out['full'], out['nosim'], rep


def analysis_agreement(text, specs):
    """Descriptive only: share of ground-truth specs found in the LLM's section for their category."""
    heads = {'fact': 'Fact Illustration', 'behavior': 'Behavior Specification', 'simple': 'Workflow Specification (Simple)',
             'complex': 'Workflow Specification (Complex)'}
    pos = sorted((text.rfind(h), c) for c, h in heads.items() if text.rfind(h) >= 0)
    sec = {c: text[p:(pos[j + 1][0] if j + 1 < len(pos) else None)] for j, (p, c) in enumerate(pos)}
    gold = {'fact': [s for s in specs if s['cat'] == 'fact'], 'behavior': [s for s in specs if s['cat'] == 'behavior'],
            'workflow': [s for s in specs if s['cat'].startswith('cond')]}

    def where(s):
        key = op_text(s['op']).rstrip('.') if 'op' in s else s['text'].splitlines()[-1][:60]
        return [c for c, t in sec.items() if key in t]
    hit = collections.Counter(); n = collections.Counter()
    for g, ss in gold.items():
        for s in ss:
            n[g] += 1; loc = where(s)
            if (g == 'workflow' and ({'simple', 'complex'} & set(loc))) or (g != 'workflow' and g in loc):
                hit[g] += 1
    return {'sections_found': sorted(sec), 'n_gold': dict(n), 'recall_by_gold_category': {g: hit[g] / n[g] for g in n if n[g]},
            'note': 'coarse substring agreement; descriptive only, not used to build data'}


def generate_raw(prompts, gen_model, seed, gpu_memory_utilization=.92):
    """Run the CPT-data prompts with a local LLM via vLLM (paper setting: Qwen3-32B, non-thinking)."""
    from vllm import LLM, SamplingParams
    llm = LLM(gen_model, max_model_len=16384, gpu_memory_utilization=gpu_memory_utilization, enable_prefix_caching=True, enforce_eager=True)
    tok = llm.get_tokenizer()
    texts = [tok.apply_chat_template([{'role': 'user', 'content': p['prompt']}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
             for p in prompts]
    sps = [SamplingParams(max_tokens=p['max_tokens'], seed=1000 * seed + j,
                          **({'temperature': p['temperature'], 'top_p': .8, 'top_k': 20} if p['temperature'] else {'temperature': 0.0}))
           for j, p in enumerate(prompts)]
    res = llm.generate(texts, sps)
    return [{**{k: v for k, v in p.items() if k != 'prompt'}, 'output': o.outputs[0].text, 'finish': o.outputs[0].finish_reason,
             'gen_model': gen_model} for p, o in zip(prompts, res)]


def read_prompt_file(path) -> str:
    return Path(path).read_text()


def dumps(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)
