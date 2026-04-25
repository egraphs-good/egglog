# autoresearch

This is an experiment to have the LLM do its own research.

## Scope

在这个 branch 里面，我实现了 index reuse。也就是说对于每一个 trie，我每次会去看我是否已经创建过它了；如果我创建过了，就不会再创建了。具体可以看我 execute.rs 里 ChildrenMap 相关的实现

这本应该是一个优化，但是看起来它并没有让我的程序的效率提高很多。所以在这一次实验中，我希望你能通过不停地深入研究我的代码来发现问题出在哪，除此之外，我还希望你能够 in general 提高我的 query execution 效率

我已经为你准备好了一个脚本，在 scripts/bench.sh 下面。你可以通过运行这个脚本来测试你的改动是否真的提高了效率。这个脚本会运行一些基准测试，并输出结果。你可以通过比较这些结果来判断你的改动是否有效。

你可以修改以下的文件

   * `core-relations/src/free_join/execute.rs`
   * `core-relations/src/free_join/plan.rs`

Other files under `core-relations/src/free_join` are also fair game, but I don't expect you to change most of the time.

**What you CAN do:**
- 修改我上面提到的文件，
你可以修改的东西包括：
1. 程序优化器
2. Index 的结构
3. Frame update
4. Trie 的结构

**What you CANNOT do:**
- 你不能将算法替换成另一种算法，比如说，你不能实现一个 hash join 来替代目前的 join algorithm。
- Install new packages or add dependencies.
- Modify the evaluation harness. (scripts/bench.sh)
- Look at files beyond the project folder `/home/yz489/egglog`.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A 1% performance improvement that adds 100 lines of hacky code? Probably not worth it. A 1% performance improvement from deleting code? Definitely keep. An improvement of ~0 but much simpler code? Keep.

**The first run**: Your very first run should always be to establish the baseline. See below.

## Logging results

你会使用我提供的 bench.sh，它提供以下的功能：
```
./scripts/bench.sh run      # build + time → append to benchmarks/result.csv
./scripts/bench.sh archive  # rename result.csv → benchmarks/2026-04-24T15:30:00.csv
./scripts/bench.sh clear    # delete result.csv
./scripts/bench.sh diff     # 对比
```

你可以运行以下命令来 set up 这个工作环境：

```
./scripts/bench.sh run      # 跑 benchmark，写入 result.csv
./scripts/bench.sh archive  # 把当前结果存档（作为 baseline）
```

然后每次你修改完代码之后，运行以下的命令：

```
./scripts/bench.sh run      # 再跑一次，覆盖 result.csv
./scripts/bench.sh diff     # 对比
```
diff 的输出示例：

```
  Comparing:
    baseline : 2026-04-24T15:30:00.csv
    current  : result.csv

  Benchmark                                  Before (s)   After (s)     Δ (s)      Δ %
  ────────────────────────────────────────────────────────────────────────────────────
  hardboiled_conv1d_32.egg                        1.234       1.100    -0.134    -10.9%  ▼ faster
  hardboiled_conv1d_128.egg                       5.678       6.100    +0.422     +7.4%  ▲ slower
  ...

  Summary: 4 faster  ·  1 slower  ·  1 unchanged  ·  0 missing
```

最后你再来选择是 archive 还是 discard 最新的结果

## The experiment loop

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune the `free_join` module with an experimental idea by directly hacking the code.
3. Once you see that it builds, git commit
4. Run the experiment: `scripts/bench.sh run 2>&1` and read the result.
5. If the grep output is empty, the run crashed. Investigate. If you can't get things to work after more than a few attempts, give up.
6. Diff the results to see if the result is improved.
7. If run time improved, you "advance" the branch, keeping the git commit. Use your judgment to decide if a run is improved. For example, if it has a huge speedup on one particular benchmark but a mild slowdown for others, this should be considered as a speedup. On the other hand, if it improves other benchmarks moderately but drastically slows down one benchmark, then I don't want that to be included.
8. In either case, record in the work journal `journal.md` what you did, what you observed, and what you think about it. This is important for keeping track of your thought process and for future reference.
9. If run time is equal or worse, you git reset back to the last commit and discard the change you made. Avoid using git revert for this purpose, as it clutters the commit history. Instead, use `git reset --hard HEAD` to go back to the last commit.

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Timeout**: Each experiment should take ~1 minute total. If a run exceeds 10 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes, re-read your work journal `journal.md`. The loop runs until the human interrupts you, period.

Other notes specific to this experiment:

- Egglog by default is run in single thread, so you don't have to worry about the parallel code paths. However, you should not introduce data structures that are known to be bad for parallelism, e.g., a global lock on some contended resources.
- Don't try to remove additions from the git history. These additions are known to be good for them to be commited.
- `journal.md` already has some lessons learned from previous experiments. Take a look before you start, and keep it updated with your own thoughts and observations.
