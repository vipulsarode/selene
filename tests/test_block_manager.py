import random

import pytest

from engine.block_manager import BlockManager


# ---------------------------------------------------------------------------
# Helper: the one rule that must ALWAYS hold, no matter what happened before.
# Every block ID lives in exactly one place: the free list OR one block table.
# ---------------------------------------------------------------------------
def check_invariant(bm: BlockManager):
    in_tables = [b for table in bm.block_tables.values() for b in table]
    everywhere = list(bm.free_blocks) + in_tables

    # free count + blocks in tables == total blocks
    assert len(everywhere) == bm.num_blocks, "a block was lost or duplicated"
    # no block in two places (two tables, or a table and the free list)
    assert len(set(everywhere)) == bm.num_blocks, "a block ID shows up twice"
    # nothing invented, nothing missing
    assert set(everywhere) == set(range(bm.num_blocks))
    # the two dicts always track the same requests
    assert bm.block_tables.keys() == bm.num_tokens.keys()


# ---------------------------------------------------------------------------
# 1. Fresh manager
# ---------------------------------------------------------------------------
def test_fresh_manager_all_blocks_free():
    bm = BlockManager(num_blocks=10)
    assert bm.num_free_blocks() == 10
    assert bm.block_tables == {}
    assert bm.num_tokens == {}
    check_invariant(bm)


# ---------------------------------------------------------------------------
# 2. Allocate
# ---------------------------------------------------------------------------
def test_allocate_40_tokens_takes_3_blocks():
    bm = BlockManager(num_blocks=10)
    table = bm.allocate("a", 40)
    assert len(table) == 3
    assert bm.num_free_blocks() == 7
    assert bm.num_tokens["a"] == 40
    assert bm.block_tables["a"] == table
    check_invariant(bm)


@pytest.mark.parametrize(
    "num_tokens, expected_blocks",
    [(1, 1), (15, 1), (16, 1), (17, 2), (32, 2), (33, 3)],
)
def test_allocate_rounds_up(num_tokens, expected_blocks):
    bm = BlockManager(num_blocks=10)
    assert len(bm.allocate("a", num_tokens)) == expected_blocks
    check_invariant(bm)


def test_two_requests_never_share_blocks():
    bm = BlockManager(num_blocks=10)
    a = bm.allocate("a", 40)
    b = bm.allocate("b", 20)
    assert set(a).isdisjoint(b)
    assert bm.num_free_blocks() == 10 - 3 - 2
    check_invariant(bm)


def test_allocate_failure_changes_nothing():
    bm = BlockManager(num_blocks=10, watermark=2)
    assert bm.allocate("a", 129) is None      # needs 9, only 8 allowed
    assert bm.num_free_blocks() == 10
    assert "a" not in bm.block_tables
    assert "a" not in bm.num_tokens
    check_invariant(bm)


# ---------------------------------------------------------------------------
# 3. append_slot
# ---------------------------------------------------------------------------
def test_append_mid_block_then_new_block():
    bm = BlockManager(num_blocks=10)
    bm.allocate("a", 40)                      # 3 blocks, last one has 8/16 used

    for _ in range(8):                        # tokens 41..48 fill the last block
        assert bm.append_slot("a") is None
        check_invariant(bm)

    new_block = bm.append_slot("a")           # token 49 needs a 4th block
    assert new_block is not None
    assert bm.block_tables["a"][-1] == new_block
    assert len(bm.block_tables["a"]) == 4
    assert bm.num_tokens["a"] == 49
    assert bm.num_free_blocks() == 6
    check_invariant(bm)


def test_append_at_exact_block_boundary():
    bm = BlockManager(num_blocks=10)
    bm.allocate("a", 32)                      # 2 blocks, both completely full
    new_block = bm.append_slot("a")           # very first append needs a new block
    assert new_block is not None
    assert len(bm.block_tables["a"]) == 3
    assert bm.num_tokens["a"] == 33
    check_invariant(bm)


def test_count_goes_up_on_every_append():
    bm = BlockManager(num_blocks=10)
    bm.allocate("a", 10)
    for i in range(1, 30):
        bm.append_slot("a")
        assert bm.num_tokens["a"] == 10 + i
    check_invariant(bm)


def test_running_request_can_dip_into_watermark():
    # The watermark only blocks NEW requests. A request that is already
    # running must still be able to grow into those reserved blocks.
    bm = BlockManager(num_blocks=10, watermark=2)
    bm.allocate("a", 128)                     # 8 blocks, 2 left (= watermark)
    assert bm.append_slot("a") is not None    # takes reserved block #1
    for _ in range(15):
        bm.append_slot("a")
    assert bm.append_slot("a") is not None    # takes reserved block #2
    assert bm.num_free_blocks() == 0
    check_invariant(bm)


# ---------------------------------------------------------------------------
# 4. free
# ---------------------------------------------------------------------------
def test_free_returns_every_block():
    bm = BlockManager(num_blocks=10)
    bm.allocate("a", 40)
    bm.allocate("b", 20)

    bm.free("a")
    assert bm.num_free_blocks() == 8
    assert "a" not in bm.block_tables
    assert "a" not in bm.num_tokens
    check_invariant(bm)

    bm.free("b")
    assert bm.num_free_blocks() == 10
    check_invariant(bm)


def test_free_after_appends_returns_grown_blocks_too():
    bm = BlockManager(num_blocks=10)
    bm.allocate("a", 16)
    for _ in range(20):                       # grows to 36 tokens = 3 blocks
        bm.append_slot("a")
    assert len(bm.block_tables["a"]) == 3
    bm.free("a")
    assert bm.num_free_blocks() == 10
    check_invariant(bm)


def test_freed_blocks_get_reused():
    bm = BlockManager(num_blocks=3)
    bm.allocate("a", 48)                      # uses all 3 blocks
    assert not bm.can_allocate(1)
    bm.free("a")
    assert bm.allocate("b", 48) is not None
    check_invariant(bm)


def test_free_unknown_request_raises():
    bm = BlockManager(num_blocks=10)
    with pytest.raises(KeyError):
        bm.free("ghost")


def test_double_free_raises_and_leaves_state_clean():
    bm = BlockManager(num_blocks=10)
    bm.allocate("a", 40)
    bm.free("a")
    with pytest.raises(KeyError):
        bm.free("a")
    assert bm.num_free_blocks() == 10         # blocks were NOT added twice
    check_invariant(bm)


# ---------------------------------------------------------------------------
# 5. Watermark / can_allocate
# ---------------------------------------------------------------------------
def test_watermark_boundary():
    bm = BlockManager(num_blocks=10, watermark=2)
    assert bm.can_allocate(128)               # 8 blocks -> leaves 2 free: OK
    assert not bm.can_allocate(129)           # 9 blocks -> leaves 1 free: no


def test_no_watermark_can_fill_everything():
    bm = BlockManager(num_blocks=10, watermark=0)
    assert bm.can_allocate(160)               # exactly 10 blocks
    assert not bm.can_allocate(161)           # 11 blocks


# ---------------------------------------------------------------------------
# 6. can_append_slot
# ---------------------------------------------------------------------------
def test_can_append_mid_block_even_with_zero_free_blocks():
    bm = BlockManager(num_blocks=3)
    bm.allocate("a", 40)                      # all 3 blocks, last one 8/16 used
    assert bm.num_free_blocks() == 0
    for _ in range(8):                        # 8 empty slots left: all fine
        assert bm.can_append_slot("a")
        bm.append_slot("a")
    assert not bm.can_append_slot("a")        # 48 tokens, last block full, no free


def test_can_append_full_block_when_free_blocks_exist():
    bm = BlockManager(num_blocks=10)
    bm.allocate("a", 32)                      # last block full, 8 free
    assert bm.can_append_slot("a")


def test_can_append_ignores_watermark():
    # Case 2 from the chat: free == watermark, last block full.
    # can_allocate(1) would say no, but a running request is allowed in.
    bm = BlockManager(num_blocks=10, watermark=2)
    bm.allocate("a", 128)                     # 8 blocks, 2 free == watermark
    assert not bm.can_allocate(1)
    assert bm.can_append_slot("a")


def test_can_append_does_not_change_state():
    bm = BlockManager(num_blocks=10)
    bm.allocate("a", 32)
    bm.can_append_slot("a")
    assert bm.num_tokens["a"] == 32
    assert bm.num_free_blocks() == 8
    check_invariant(bm)


def test_can_append_unknown_request_raises():
    bm = BlockManager(num_blocks=10)
    with pytest.raises(KeyError):
        bm.can_append_slot("ghost")


def test_preempt_newest_frees_room_for_oldest():
    # The scheduler flow for tomorrow: A is oldest, B newest, memory is full.
    bm = BlockManager(num_blocks=4)
    bm.allocate("A", 32)                      # 2 full blocks
    bm.allocate("B", 32)                      # 2 full blocks, 0 free
    assert not bm.can_append_slot("A")

    b_tokens = bm.num_tokens["B"]             # remember B's progress
    bm.free("B")                              # preempt B (recompute later)
    assert bm.can_append_slot("A")
    assert bm.append_slot("A") is not None
    check_invariant(bm)

    bm.free("A")                              # A finishes
    assert bm.allocate("B", b_tokens) is not None   # B re-prefills
    check_invariant(bm)


# ---------------------------------------------------------------------------
# 7. Stress test: random mix of admit / decode / finish, with recompute
#    preemption when memory runs out. Invariant checked after every step.
# ---------------------------------------------------------------------------
def test_random_workload_with_preemption_keeps_invariant():
    rng = random.Random(0)
    bm = BlockManager(num_blocks=16, watermark=2)
    waiting = []          # [(req_id, num_tokens)], front = next to admit
    running = []          # oldest first, newest last
    next_id = 0
    preemptions = 0

    for _ in range(5000):
        action = rng.random()

        if action < 0.2:                                      # new request arrives
            waiting.append((next_id, rng.randint(1, 60)))
            next_id += 1

        elif action < 0.9 and running:                        # decode one token
            req = rng.choice(running)
            while not bm.can_append_slot(req):
                victim = running.pop()                        # newest request
                waiting.insert(0, (victim, bm.num_tokens[victim]))
                bm.free(victim)
                preemptions += 1
                check_invariant(bm)
                if victim == req:
                    break
            else:
                bm.append_slot(req)                           # never crashes here

        elif running:                                         # request finishes
            bm.free(running.pop(rng.randrange(len(running))))

        # admit from the front of the queue while there's room
        while waiting and bm.can_allocate(waiting[0][1]):
            req_id, n = waiting.pop(0)
            bm.allocate(req_id, n)
            running.append(req_id)

        check_invariant(bm)

    assert preemptions > 0, "test never hit memory pressure"
    for req in running:
        bm.free(req)
    assert bm.num_free_blocks() == 16
    check_invariant(bm)