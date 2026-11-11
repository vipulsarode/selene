import random

import pytest

from engine.block_manager import BlockManager
from engine.scheduler import Request, Scheduler, SchedulerOutput, Status


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_req(req_id, prompt_len, max_tokens=8, eos=None):
    return Request(req_id=req_id, prompt_token_ids=list(range(prompt_len)),
                   max_tokens=max_tokens, eos_token_id=eos)


def fake_tokens(out, token=1):
    """Pretend to be the model runner: one token per request that ran."""
    return {r.req_id: token for r in out.prefill + out.decode}


def check_scheduler_invariants(s: Scheduler):
    running_ids = [r.req_id for r in s.running]
    waiting_ids = [r.req_id for r in s.waiting]
    # nobody is in two places
    assert len(set(running_ids)) == len(running_ids)
    assert not set(running_ids) & set(waiting_ids), "request in running AND waiting"
    # statuses match the list each request sits in
    assert all(r.status == Status.RUNNING for r in s.running)
    assert all(r.status == Status.WAITING for r in s.waiting)
    # only running requests own blocks
    assert set(s.bm.block_tables.keys()) == set(running_ids)
    # cap respected
    assert len(s.running) <= s.max_num_seqs
    # every block is in exactly one place
    in_tables = [b for t in s.bm.block_tables.values() for b in t]
    everywhere = list(s.bm.free_blocks) + in_tables
    assert sorted(everywhere) == list(range(s.bm.num_blocks))


def check_kv_lags_by_one(s: Scheduler):
    """After update(): the cache holds one token fewer than prompt + output,
    because the newest token hasn't been fed in yet."""
    for r in s.running:
        assert s.bm.num_tokens[r.req_id] == r.num_tokens - 1


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------
def test_num_tokens_counts_prompt_and_output():
    r = make_req("a", 5)
    assert r.num_tokens == 5
    r.output_token_ids += [7, 8]
    assert r.num_tokens == 7


def test_not_finished_before_first_token():
    assert not make_req("a", 5, max_tokens=1, eos=0).is_finished()


def test_finished_on_max_tokens_counts_output_only():
    r = make_req("a", 100, max_tokens=3)      # long prompt must NOT count
    r.output_token_ids = [1, 1]
    assert not r.is_finished()
    r.output_token_ids.append(1)
    assert r.is_finished()


def test_finished_on_eos():
    r = make_req("a", 5, max_tokens=50, eos=99)
    r.output_token_ids = [1, 2]
    assert not r.is_finished()
    r.output_token_ids.append(99)
    assert r.is_finished()


def test_is_finished_has_no_side_effects():
    r = make_req("a", 5, max_tokens=1)
    r.output_token_ids = [1]
    r.status = Status.RUNNING
    assert r.is_finished()
    assert r.status == Status.RUNNING


# ---------------------------------------------------------------------------
# add_request / has_unfinished
# ---------------------------------------------------------------------------
def test_add_request_goes_to_back():
    s = Scheduler(BlockManager(10))
    s.add_request(make_req("a", 5))
    s.add_request(make_req("b", 5))
    assert [r.req_id for r in s.waiting] == ["a", "b"]


def test_has_unfinished():
    s = Scheduler(BlockManager(10))
    assert not s.has_unfinished()
    s.add_request(make_req("a", 5))
    assert s.has_unfinished()


# ---------------------------------------------------------------------------
# Prefill half
# ---------------------------------------------------------------------------
def test_admits_everyone_that_fits_in_one_step():
    s = Scheduler(BlockManager(10))
    for i in range(3):
        s.add_request(make_req(f"r{i}", 16))      # 1 block each
    out = s.schedule()
    assert [r.req_id for r in out.prefill] == ["r0", "r1", "r2"]
    assert out.decode == [] and out.preempted == []
    assert all(r.status == Status.RUNNING for r in out.prefill)
    assert s.bm.num_free_blocks() == 7
    check_scheduler_invariants(s)


def test_respects_max_num_seqs():
    s = Scheduler(BlockManager(100), max_num_seqs=2)
    for i in range(5):
        s.add_request(make_req(f"r{i}", 4))
    out = s.schedule()
    assert len(out.prefill) == 2
    assert len(s.waiting) == 3
    check_scheduler_invariants(s)


def test_no_skipping_ahead():
    s = Scheduler(BlockManager(4))
    s.add_request(make_req("big", 16 * 5))        # needs 5 blocks, never fits now
    s.add_request(make_req("small", 4))           # would fit, but must wait its turn
    out = s.schedule()
    assert out.prefill == []
    assert [r.req_id for r in s.waiting] == ["big", "small"]


def test_allocates_for_num_tokens_not_just_prompt():
    s = Scheduler(BlockManager(10))
    r = make_req("a", 16)
    r.output_token_ids = [1] * 16                 # e.g. a preempted request coming back
    s.add_request(r)
    s.schedule()
    assert len(s.bm.block_tables["a"]) == 2
    assert s.bm.num_tokens["a"] == 32


# ---------------------------------------------------------------------------
# The starvation bug from Monday night
# ---------------------------------------------------------------------------
def test_running_requests_still_decode_when_front_of_waiting_does_not_fit():
    s = Scheduler(BlockManager(10))
    s.add_request(make_req("a", 60, max_tokens=50))   # 4 blocks
    s.add_request(make_req("b", 60, max_tokens=50))   # 4 blocks
    out = s.schedule()
    s.update(out, fake_tokens(out))
    s.add_request(make_req("c", 16 * 5))              # needs 5, only 2 free
    out = s.schedule()
    assert out.prefill == []
    assert [r.req_id for r in out.decode] == ["a", "b"]   # NOT an empty step
    check_scheduler_invariants(s)


# ---------------------------------------------------------------------------
# Decode half
# ---------------------------------------------------------------------------
def test_decode_reserves_one_slot_per_request_oldest_first():
    s = Scheduler(BlockManager(10))
    s.add_request(make_req("a", 5))
    s.add_request(make_req("b", 5))
    out = s.schedule()
    s.update(out, fake_tokens(out))
    check_kv_lags_by_one(s)
    out = s.schedule()
    assert [r.req_id for r in out.decode] == ["a", "b"]
    assert s.bm.num_tokens["a"] == 6                  # the slot for the token being fed in
    s.update(out, fake_tokens(out))
    check_kv_lags_by_one(s)
    check_scheduler_invariants(s)


def test_preempts_newest_to_make_room_for_oldest():
    s = Scheduler(BlockManager(4))
    s.add_request(make_req("A", 32, max_tokens=50))   # 2 full blocks
    s.add_request(make_req("B", 32, max_tokens=50))   # 2 full blocks -> cache full
    out = s.schedule()
    s.update(out, fake_tokens(out))                   # each next token needs a 3rd block
    out = s.schedule()
    assert [r.req_id for r in out.preempted] == ["B"]
    assert [r.req_id for r in out.decode] == ["A"]
    b = s.waiting[0]
    assert b.req_id == "B" and b.status == Status.WAITING
    assert b.output_token_ids == [1]                  # tokens kept
    assert "B" not in s.bm.block_tables              # blocks given back
    check_scheduler_invariants(s)


def test_preempted_request_jumps_ahead_of_new_arrivals():
    s = Scheduler(BlockManager(4))
    s.add_request(make_req("A", 32, max_tokens=50))
    s.add_request(make_req("B", 32, max_tokens=50))
    out = s.schedule(); s.update(out, fake_tokens(out))
    s.add_request(make_req("C", 16))                  # arrives later, has to wait
    out = s.schedule()                                # B gets preempted
    assert [r.req_id for r in s.waiting] == ["B", "C"]   # B first: it was here before C


def test_newest_preempts_itself_and_older_one_still_decodes():
    s = Scheduler(BlockManager(4))
    s.add_request(make_req("A", 20, max_tokens=50))   # 2 blocks, room left in last one
    s.add_request(make_req("B", 32, max_tokens=50))   # 2 full blocks -> cache full
    out = s.schedule()
    s.update(out, fake_tokens(out))                   # A has room, B needs a 3rd block
    out = s.schedule()
    assert [r.req_id for r in out.decode] == ["A"]
    assert [r.req_id for r in out.preempted] == ["B"]
    assert s.waiting[0].req_id == "B"
    check_scheduler_invariants(s)


def test_lone_request_preempting_itself_does_not_crash():
    s = Scheduler(BlockManager(2), max_num_seqs=1)
    s.add_request(make_req("A", 32, max_tokens=50))   # fills the whole cache
    out = s.schedule()
    s.update(out, fake_tokens(out))
    out = s.schedule()                                # needs a 3rd block that doesn't exist
    assert out.decode == []
    assert [r.req_id for r in out.preempted] == ["A"]
    assert s.running == []
    check_scheduler_invariants(s)


def test_preempted_request_resumes_with_its_tokens():
    s = Scheduler(BlockManager(4))
    s.add_request(make_req("A", 32, max_tokens=3))
    s.add_request(make_req("B", 32, max_tokens=50))
    out = s.schedule(); s.update(out, fake_tokens(out))       # prefill both
    out = s.schedule(); s.update(out, fake_tokens(out))       # B preempted, A decodes
    out = s.schedule(); s.update(out, fake_tokens(out))       # A finishes
    assert "A" not in [r.req_id for r in s.running]
    out = s.schedule()                                        # B comes back
    assert [r.req_id for r in out.prefill] == ["B"]
    assert s.bm.num_tokens["B"] == 33                         # prompt + its 1 kept token
    check_scheduler_invariants(s)


# ---------------------------------------------------------------------------
# update()
# ---------------------------------------------------------------------------
def test_update_gives_prefill_requests_their_first_token():
    s = Scheduler(BlockManager(10))
    s.add_request(make_req("a", 5))
    out = s.schedule()
    s.update(out, {"a": 42})
    assert s.running[0].output_token_ids == [42]


def test_update_retires_finished_requests():
    s = Scheduler(BlockManager(10))
    s.add_request(make_req("a", 5, max_tokens=2))
    s.add_request(make_req("b", 5, max_tokens=50, eos=99))
    out = s.schedule(); s.update(out, {"a": 1, "b": 1})
    out = s.schedule()
    done = s.update(out, {"a": 1, "b": 99})          # a hits max_tokens, b hits EOS
    assert sorted(r.req_id for r in done) == ["a", "b"]
    assert all(r.status == Status.FINISHED for r in done)
    assert s.running == [] and not s.has_unfinished()
    assert s.bm.num_free_blocks() == 10
    check_scheduler_invariants(s)


# ---------------------------------------------------------------------------
# End to end: random workload, invariants after every step
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_random_workload_every_request_finishes(seed):
    rng = random.Random(seed)
    s = Scheduler(BlockManager(num_blocks=12, watermark=1), max_num_seqs=6)
    expected, finished = {}, {}
    preemptions = 0

    for i in range(30):
        n_out = rng.randint(20, 60)
        r = make_req(f"r{i}", rng.randint(1, 60), max_tokens=n_out)
        expected[r.req_id] = n_out
        s.add_request(r)

    for _ in range(10_000):
        if not s.has_unfinished():
            break
        out = s.schedule()
        preemptions += len(out.preempted)
        assert not (out.prefill and out.decode), "v1 mixes prefill and decode"
        for r in s.update(out, fake_tokens(out)):
            finished[r.req_id] = len(r.output_token_ids)
        check_scheduler_invariants(s)
        check_kv_lags_by_one(s)

    assert not s.has_unfinished(), "engine got stuck"
    assert finished == expected            # every request got exactly max_tokens
    assert preemptions > 0, "workload never hit memory pressure"
    assert s.bm.num_free_blocks() == 12