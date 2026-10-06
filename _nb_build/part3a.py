from common import md, code


def cells():
    return [
        md(
            """
# Part 3 — RAG-relevant DSA

You already know the algorithms. This section is **Python + why a RAG engineer uses it**.

For each problem: statement, intuition, brute force, optimal, implementation, complexities, mistakes, edges, hidden tests, RAG why, then runnable cases.

---

## Phase 1 — Python + hashing
"""
        ),
        md(
            """
### 3.1 Two Sum

**Statement.** `nums` and `target`. Return indices `i, j` (`i != j`) with `nums[i] + nums[j] == target`.

**Intuition.** For each value `x`, you want `target - x` already seen. That is a dict: value → index.

**Brute.** Double loop O(n²).

**Optimal.** One pass hash map O(n) time, O(n) space.

**RAG why.** "Find the stored vector that pairs with this query" is the same lookup shape: key → payload. Also: joining `chunk_id` to a row.
"""
        ),
        code(
            """
def two_sum(nums, target):
    seen = {}
    for i, value in enumerate(nums):
        need = target - value
        if need in seen:
            return [seen[need], i]
        seen[value] = i
    return []


def _check_two_sum():
    cases = [
        ([2, 7, 11, 15], 9, [0, 1]),
        ([3, 2, 4], 6, [1, 2]),
        ([3, 3], 6, [0, 1]),          # duplicates
        ([], 0, []),
        ([1], 1, []),
        ([-1, -2, -3], -5, [1, 2]),   # negatives
        ([0, 4, 3, 0], 0, [0, 3]),
        ([10**9, 1, 10**9], 2 * 10**9, [0, 2]),
        ([1, 2, 3], 7, []),
    ]
    for nums, target, expected in cases:
        got = two_sum(nums, target)
        if expected == []:
            assert got == []
        else:
            i, j = got
            assert nums[i] + nums[j] == target
    print("two_sum ok")


_check_two_sum()
"""
        ),
        md(
            """
**Mistakes:** storing value→value instead of value→index; using the same index twice (`need == value` and `seen[value] == i`); `seen[value] = i` **before** the lookup when duplicates must pair.

**Hidden:** two zeros; `target` overflow if you think Python ints are 32-bit (they are not).

**Takeaway:** `enumerate` + dict lookup.
"""
        ),
        md(
            """
### 3.2 Contains Duplicate

**Statement.** True iff any value appears twice.

**Optimal.** `return len(nums) != len(set(nums))` or a growing set (early exit).

**RAG why.** Duplicate document IDs, duplicate chunk IDs, ingesting the same file twice.
"""
        ),
        code(
            """
def contains_duplicate(nums):
    seen = set()
    for x in nums:
        if x in seen:
            return True
        seen.add(x)
    return False


assert contains_duplicate([1, 2, 3, 1]) is True
assert contains_duplicate([1, 2, 3, 4]) is False
assert contains_duplicate([]) is False
assert contains_duplicate([0]) is False
assert contains_duplicate([0, 0]) is True
assert contains_duplicate([-1, -1]) is True
assert contains_duplicate([10**12, 10**12 + 1]) is False
print("contains_duplicate ok")
"""
        ),
        md(
            """
**Mistake:** sorting in-place when the caller still needs original order. **Complexity:** O(n) / O(n). Sorting is O(n log n) and is worse unless you must not use extra memory.
"""
        ),
        md(
            """
### 3.3 Valid Anagram

**Statement.** Are `s` and `t` anagrams?

**Optimal.** `Counter(s) == Counter(t)` after length check. Or 26-slot array if only lowercase English.

**RAG why.** Token-set similarity, near-duplicate detection, "same bag of words after normalize".
"""
        ),
        code(
            """
def valid_anagram(s, t):
    if len(s) != len(t):
        return False
    freq = {}
    for ch in s:
        freq[ch] = freq.get(ch, 0) + 1
    for ch in t:
        if ch not in freq:
            return False
        freq[ch] -= 1
        if freq[ch] == 0:
            del freq[ch]
    return not freq


assert valid_anagram("anagram", "nagaram") is True
assert valid_anagram("rat", "car") is False
assert valid_anagram("", "") is True
assert valid_anagram("a", "a") is True
assert valid_anagram("ab", "a") is False
assert valid_anagram("aacc", "ccac") is False
assert valid_anagram("Aa", "aA") is True   # case-sensitive here
assert valid_anagram("你好", "好你") is True
print("valid_anagram ok")
"""
        ),
        md(
            """
**Mistakes:** `sorted(s) == sorted(t)` is correct but O(n log n); lowercasing when the prompt is case-sensitive; using `set(s) == set(t)` (ignores counts).
"""
        ),
        md(
            """
### 3.4 Group Anagrams

**Statement.** Group strings that are anagrams.

**Optimal.** Key = `tuple(sorted(s))` or a 26-count tuple. `defaultdict(list)`.

**RAG why.** Near-duplicate chunks, grouping filenames, clustering after a cheap lexical key before you embed.
"""
        ),
        code(
            """
def group_anagrams(strs):
    groups = defaultdict(list)
    for s in strs:
        groups[tuple(sorted(s))].append(s)
    return list(groups.values())


def _norm(groups):
    return sorted(sorted(g) for g in groups)


assert _norm(group_anagrams(["eat", "tea", "tan", "ate", "nat", "bat"])) == _norm(
    [["eat", "tea", "ate"], ["tan", "nat"], ["bat"]]
)
assert group_anagrams([""]) == [[""]]
assert group_anagrams(["a"]) == [["a"]]
assert _norm(group_anagrams(["ddddddddddg", "dgggggggggg"])) != _norm([["ddddddddddg", "dgggggggggg"]])
print("group_anagrams ok")
"""
        ),
        md(
            """
**Mistakes:** using `str(sorted(s))` (ugly but works); using `sorted(s)` as a dict key (`TypeError: unhashable`); mutating the input list.

**Hidden:** empty string group; unicode; 10k identical strings.
"""
        ),
        md(
            """
### 3.5 First Unique Character

**Statement.** Index of the first non-repeating character, else -1.

**Optimal.** Count, then second pass left-to-right.

**RAG why.** Deterministic tie-break, "first document that is not a duplicate", stream of IDs.
"""
        ),
        code(
            """
def first_unique_char(s):
    counts = Counter(s)
    for i, ch in enumerate(s):
        if counts[ch] == 1:
            return i
    return -1


assert first_unique_char("leetcode") == 0
assert first_unique_char("loveleetcode") == 2
assert first_unique_char("aabb") == -1
assert first_unique_char("") == -1
assert first_unique_char("z") == 0
assert first_unique_char("cc") == -1
assert first_unique_char("aab") == 2
print("first_unique_char ok")
"""
        ),
        md(
            """
---

## Phase 2 — Text / window processing

These three problems are the mental model for **chunking and overlap**.
"""
        ),
        md(
            """
### 3.6 Longest Substring Without Repeating Characters

**Statement.** Length of the longest substring with all unique characters.

**Intuition.** Sliding window `[left, right]` + set/map of chars in the window. When a duplicate arrives, advance `left`.

**RAG why.** Deduping tokens inside a window; understanding why overlap windows move by `step = size - overlap`.
"""
        ),
        code(
            """
def length_of_longest_substring(s):
    last = {}
    left = 0
    best = 0
    for right, ch in enumerate(s):
        if ch in last and last[ch] >= left:
            left = last[ch] + 1
        last[ch] = right
        best = max(best, right - left + 1)
    return best


assert length_of_longest_substring("abcabcbb") == 3
assert length_of_longest_substring("bbbbb") == 1
assert length_of_longest_substring("pwwkew") == 3
assert length_of_longest_substring("") == 0
assert length_of_longest_substring(" ") == 1
assert length_of_longest_substring("au") == 2
assert length_of_longest_substring("dvdf") == 3
assert length_of_longest_substring("abba") == 2
print("longest_substring ok")
"""
        ),
        md(
            """
**Mistakes:** forgetting to only move `left` forward (`abba`); using a set and `while` but forgetting to delete `s[left]`.

**Complexity:** O(n) time, O(min(n, alphabet)) space.
"""
        ),
        md(
            """
### 3.7 Minimum Window Substring

**Statement.** Smallest substring of `s` that covers every character in `t` (including duplicates).

**Intuition.** Expand right until the window is valid, then shrink left. Need a "how many unique constraints satisfied" counter.

**RAG why.** This is **coverage**: "smallest chunk that contains all query terms". Also the pattern for "grow until enough context, then shrink".
"""
        ),
        code(
            """
def min_window(s, t):
    if not t or len(t) > len(s):
        return ""
    need = Counter(t)
    missing = len(need)
    left = 0
    best = (0, float("inf"))
    window = {}
    for right, ch in enumerate(s):
        window[ch] = window.get(ch, 0) + 1
        if ch in need and window[ch] == need[ch]:
            missing -= 1
        while missing == 0:
            if right - left < best[1] - best[0]:
                best = (left, right)
            left_ch = s[left]
            window[left_ch] -= 1
            if left_ch in need and window[left_ch] < need[left_ch]:
                missing += 1
            left += 1
    if best[1] is float("inf"):
        return ""
    i, j = best
    return s[i : j + 1]


assert min_window("ADOBECODEBANC", "ABC") == "BANC"
assert min_window("a", "a") == "a"
assert min_window("a", "aa") == ""
assert min_window("", "a") == ""
assert min_window("a", "") == ""
assert min_window("aa", "aa") == "aa"
assert min_window("bba", "ab") == "ba"
print("min_window ok")
"""
        ),
        md(
            """
**Mistakes:** treating `t` as a set (drops duplicate requirements); off-by-one on the stored bounds; shrinking before the window is valid.

**Hidden:** `t` longer than `s`; all characters the same; Unicode.
"""
        ),
        md(
            """
### 3.8 Merge Intervals

**Statement.** Merge overlapping `[start, end]` intervals.

**Intuition.** Sort by start. If `cur.start <= prev.end`, merge ends.

**RAG why.** Overlapping chunks, merging highlight ranges, collapsing adjacent passages before sending context to an LLM, detecting that `overlap >= chunk_size` never advances.
"""
        ),
        code(
            """
def merge_intervals(intervals):
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda iv: iv[0])
    merged = [intervals[0][:]]
    for start, end in intervals[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


assert merge_intervals([[1, 3], [2, 6], [8, 10], [15, 18]]) == [[1, 6], [8, 10], [15, 18]]
assert merge_intervals([[1, 4], [4, 5]]) == [[1, 5]]      # touching
assert merge_intervals([]) == []
assert merge_intervals([[1, 1]]) == [[1, 1]]
assert merge_intervals([[1, 4], [2, 3]]) == [[1, 4]]      # contained
assert merge_intervals([[5, 6], [1, 2]]) == [[1, 2], [5, 6]]
assert merge_intervals([[1, 10], [2, 3], [4, 8]]) == [[1, 10]]
print("merge_intervals ok")
"""
        ),
        md(
            """
**Mistakes:** not sorting; `start < prev.end` instead of `<=` (depends on whether touching merges — usually yes); mutating the input without copying.

**Chunking connection.** A word window of size 4 overlap 1 is intervals `[0,4), [3,7), [6,10)`. Merge if you need a single span for citation.
"""
        ),
    ]
