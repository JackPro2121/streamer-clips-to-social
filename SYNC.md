# SYNC.md — keeping `core/` aligned with `clips-core`

`core/` in this repository is a **vendored copy** of the shared engine
[`JackPro2121/clips-core`](https://github.com/JackPro2121/clips-core): ordinary
committed files, not a submodule. Nothing needs initialising after a clone.

Because the copy is vendored, upstream improvements do not arrive on their own.
This file records exactly which upstream revision was copied and how to check
for, and pull in, later changes.

> This file lives at the repository root on purpose. `core/` is kept
> byte-identical to the upstream tree so that `git diff` against `clips-core`
> stays clean — adding anything inside `core/` would break that.

## Provenance

| | |
| --- | --- |
| Upstream | `https://github.com/JackPro2121/clips-core.git` |
| Copied from | commit `4fc1c24bd5ed3a837c8a8d89f4b97299cbf77da1` |
| Copied tree | `fef54c3ff74097777d8c314fb5fef7aba2fa5383` |
| Local path | `core/` |
| Tracked files | 49 |

`clips-core`'s repository root **is** the `core` package, so upstream's commit
tree hash and this repo's `core/` tree hash are directly comparable — that single
comparison is the whole drift check.

## Drift check

No setup and no permanent remote required:

```bash
# pull upstream's latest main into FETCH_HEAD without adding a remote
git fetch https://github.com/JackPro2121/clips-core.git main

# 1) quick verdict
git rev-parse HEAD:core            # our vendored copy
git rev-parse 'FETCH_HEAD^{tree}'  # upstream clips-core
#   equal   -> in sync, nothing to do
#   unequal -> drifted, continue to step 2

# 2) see exactly what changed upstream since the copy
git diff FETCH_HEAD HEAD:core
```

`git diff FETCH_HEAD HEAD:core` compares the two trees directly, so it works even
though upstream's files sit at a repository root while ours sit under `core/`.
Empty output means byte-for-byte identical.

## Pulling upstream changes in

```bash
git fetch https://github.com/JackPro2121/clips-core.git main

# review first
git diff FETCH_HEAD HEAD:core

# replace the vendored copy with upstream's tree, exactly
git rm -r --quiet core
git read-tree --prefix=core/ -u FETCH_HEAD
git add -A core

# confirm the copy now matches upstream (must print nothing)
git diff FETCH_HEAD HEAD:core

git commit -m "chore(core): sync vendored engine with clips-core <short-sha>"
```

`git rm -r core` removes only the **tracked** engine files, so locally
downloaded model weights under `core/models/` (SFace, SyncNet, TransNetV2 — all
gitignored) survive the sync. `git read-tree --prefix=core/ -u FETCH_HEAD`
unpacks upstream's tree at that path in a single step: it removes files upstream
deleted as well as adding and updating the rest.

<details>
<summary>Alternative that uses <code>tar</code> instead of <code>read-tree</code></summary>

```bash
git rm -r --quiet core
mkdir -p core
git archive FETCH_HEAD | tar -x -C core
git add -A core
```

Equivalent result. Needs `tar` (present on Linux, macOS and Windows 10+) and a
binary-safe shell — prefer the `read-tree` form on Windows PowerShell.
</details>

## Contributing a fix back upstream

`core/` is deliberately editable here, but if a change fixes the engine itself
rather than V2, land it upstream too so the V1 podcast pipeline receives it:

```bash
git clone https://github.com/JackPro2121/clips-core.git
cd clips-core
# make the change, add tests, commit, push
```

Then sync it back into this repo with the procedure above and update the
**Provenance** table.

## Notes

* V1 (`podcasts-clips-to-social`) keeps its own `core/` mount; a change made here
  does not reach it, and vice versa.
* `core/README.md` is the upstream engine README and still describes the
  standalone `clips-core` repo and its submodule usage. It is left untouched on
  purpose so the vendored tree stays byte-identical to upstream.
