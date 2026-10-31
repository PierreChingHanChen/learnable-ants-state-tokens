"""Learnable Ant 2 的單元測試：語彙格式、世界機制、教師與 dual 核心。"""
import numpy as np
import pytest

from miat_ants.la2 import vocab as V, metrics as M
from miat_ants.la2.system import LA2Config, LA2System
from miat_ants.la2.world import World2, two_patch
from miat_ants.core.field_medium import FieldMedium


def test_vocab_disjoint_and_size():
    assert V.V == 1 + 10 + 25 + 8
    assert not (V.ACTION_MASK & V.OBS_MASK).any()
    assert not (V.STATE_MASK & (V.ACTION_MASK | V.OBS_MASK)).any()


def test_teacher_sequences_valid_dual_and_bo():
    r = LA2System(LA2Config(map_name="two_patch", core="teacher", seed=3, max_steps=300,
                            collect_episodes=True)).run()
    assert r.episodes
    for e in r.episodes:
        V.check_sequence(e, "dual")
        V.check_sequence(V.to_bo(e), "bo")


def test_take_sets_memory_and_give_up_clears():
    spec = two_patch()
    med = FieldMedium(*spec.shape, 1, rng=np.random.default_rng(0))
    w = World2(spec, 1, med, np.random.default_rng(0), give_up=3)
    w.pos[0] = (5, 5)
    w.apply(np.array([0]), np.array([V.STOI["T"]]))
    assert w.carrying[0] and w.has_mem[0] and tuple(w.mem_pos[0]) == (5, 5) and w.carry_q[0] == 1
    w.carrying[0] = False
    w.food[3:8, 3:8] = 0                      # 食物消失
    for _ in range(6):
        w.apply(np.array([0]), np.array([V.STOI["K"]]))
    assert not w.has_mem[0]


def test_uturn_and_scan_actions():
    spec = two_patch()
    med = FieldMedium(*spec.shape, 1, rng=np.random.default_rng(0))
    w = World2(spec, 1, med, np.random.default_rng(0))
    w.pos[0] = (16, 10); w.heading[0] = 0
    w.apply(np.array([0]), np.array([V.STOI["U"]]))
    assert w.heading[0] == 4 and tuple(w.pos[0]) == (16, 9)
    w.apply(np.array([0]), np.array([V.STOI["K"]]))
    assert tuple(w.pos[0]) == (16, 9) and w.heading[0] in (2, 6)


def test_goal_tokens_visible_with_memory():
    spec = two_patch()
    med = FieldMedium(*spec.shape, 1, rng=np.random.default_rng(0))
    w = World2(spec, 1, med, np.random.default_rng(0))
    w.has_mem[0] = True; w.mem_pos[0] = (16, 28); w.pos[0] = (16, 16); w.heading[0] = 0
    assert V.ITOS[int(w.perceive(np.array([0]))[0])] == "g"
    w.heading[0] = 4
    assert V.ITOS[int(w.perceive(np.array([0]))[0])] == "j"


def test_dual_gpt_run_produces_dual_sequences():
    r = LA2System(LA2Config(map_name="two_patch", core="gpt", mode="dual", seed=1, max_steps=100,
                            n_ants=5, collect_episodes=True, base_weights="")).run()
    for e in r.episodes:
        V.check_sequence(e, "dual")


def test_ngram_and_entropy_on_teacher():
    r = LA2System(LA2Config(map_name="two_patch", core="teacher", seed=5, max_steps=300,
                            collect_episodes=True)).run()
    eps = r.episodes
    assert M.cond_entropy(eps, "s_prev,o") <= M.cond_entropy(eps, "o") + 1e-9
    acc = M.eval_ngram(eps, eps, 4)["all"]
    assert 0.5 < acc <= 1.0


# ---------------------------------------------------------------- aux 格式、取樣、GRU、濾波器、測試環境
def test_aux_view_aligns_states_with_perceptions():
    r = LA2System(LA2Config(map_name="two_patch", core="teacher", seed=4, max_steps=200, collect_episodes=True)).run()
    e = r.episodes[0]
    bo, lab = V.aux_view(e)
    assert bo == V.to_bo(e)
    O, S, A = M.steps_of(e)
    assert [l for l in lab if l >= 0] == list(S)
    assert all(V.OBS_MASK[bo[i]] for i, l in enumerate(lab) if l >= 0)


def test_sample_windows_sigma_weighting_includes_sigma():
    r = LA2System(LA2Config(map_name="two_patch", core="teacher", seed=5, max_steps=800, collect_episodes=True)).run()
    eps = r.episodes
    rng = np.random.default_rng(0)
    for mode in ("dual", "aux"):
        _, heads, pr_sig = V.sample_windows(eps, 200, 48 if mode == "dual" else 32, rng, mode, "sigma", sig_frac=1.0)
        _, _, pr_trip = V.sample_windows(eps, 200, 48 if mode == "dual" else 32, rng, mode, "trip")
        assert pr_sig > pr_trip


def test_forward_last_matches_forward():
    from miat_ants.microgpt import MicroGPT, GPTConfig
    m = MicroGPT(GPTConfig(V.V, 48, 16), n_agents=3, seed=1)
    for k in m.p:
        m.p[k] = m.p[k] + np.random.default_rng(2).normal(0, 0.1, m.p[k].shape)
    idx = np.random.default_rng(3).integers(0, V.V, (3, 4, 20))
    assert np.allclose(m.forward_last(idx), m.forward(idx)[:, :, -1], atol=1e-12)


def test_gru_gradient_matches_finite_difference():
    from miat_ants.la2 import gru as G
    r = LA2System(LA2Config(map_name="two_patch", core="teacher", seed=6, max_steps=150, collect_episodes=True)).run()
    m = G.GRUModel(seed=0)
    b = G.make_batch(r.episodes[:2])
    for obj in ("bo", "aux"):
        _, g = G.loss_grad(m, b, obj)
        o = m.p["wh"][3, 7]
        m.p["wh"][3, 7] = o + 1e-5; lp, _ = G.loss_grad(m, b, obj)
        m.p["wh"][3, 7] = o - 1e-5; lm, _ = G.loss_grad(m, b, obj)
        m.p["wh"][3, 7] = o
        assert abs((lp - lm) / 2e-5 - g["wh"][3, 7]) < 1e-6


def test_bayes_filter_tracks_carrying_states():
    r = LA2System(LA2Config(map_name="two_patch", core="teacher", seed=7, max_steps=600, collect_episodes=True)).run()
    f = M.BayesStateFilter().fit(r.episodes)
    e = r.episodes[-1]
    pred = f.predict_ep(e)
    S = M.steps_of(e)[1]
    carry = np.isin(S, [V.STOI["η"], V.STOI["δ"]])
    assert (pred[carry] == S[carry]).all()


def test_aux_colony_records_dual_format_and_does_not_feed_back():
    from miat_ants.la2 import pretrain as P
    r = LA2System(LA2Config(map_name="two_patch", core="teacher", seed=8, max_steps=300, collect_episodes=True)).run()
    m, _ = P.train("aux", r.episodes, r.episodes[:20], train_steps=5, seed=0)
    import tempfile, os
    path = os.path.join(tempfile.mkdtemp(), "aux.npz"); m.save(path)
    sysm = LA2System(LA2Config(map_name="two_patch", core="gpt", mode="aux", base_weights=path, learn=True, seed=1,
                               max_steps=100, collect_episodes=True))
    rr = sysm.run()
    for k in range(5):
        V.check_sequence(sysm.ctx[k], "dual")            # 說出的狀態以 dual 格式記錄
    c = rr.core._model_ctx(sysm.ctx[0])
    assert not any(V.STATE_MASK[t] for t in c)             # 但模型的輸入不含狀態詞
    assert c == V.to_bo(sysm.ctx[0])[-rr.core.block:]


def test_barrier_map_respawn_avoids_walls():
    r = LA2System(LA2Config(map_name="barrier", core="teacher", seed=2, max_steps=1500)).run()
    w = r.world
    assert not (w.walls & (w.food > 0)).any()
