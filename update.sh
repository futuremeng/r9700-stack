#!/bin/sh
# Update this repository without touching what is expensive: the weights in ~/models, the launch caches
# (~/vllmstock-cache-*, ~/vllm-native-*), the image unless its pin moved, and any launch that is running.
# What it does: git pull this checkout, then what ./setup.sh checks -- the pinned image (rebuild offered when
# docker/Dockerfile's BASE changed), the SECOND checkout serve.sh mounts ($HOME/r9700-build/repo, which is where
# the running plugin actually comes from), the host tests, and then it offers a launch. Start one later with
# ./setup.sh. Options are passed on to setup.py.
cd "$(dirname "$0")" || exit 1
main() {
  if [ ! -d .git ]; then
    echo "This copy was not made with git, so it cannot fetch its own files. Download the newest one:"
    echo "  git clone https://github.com/futuremeng/r9700-stack && cd r9700-stack && ./setup.sh"
    echo "Your weights and caches live outside the repository and are not touched either way."
  elif ! command -v git >/dev/null 2>&1; then
    echo "This folder is a git clone but git is not installed: install it (sudo apt install git) or run"
    echo "\"git pull\" here yourself, then run ./update.sh again."
    exit 1
  else
    # what would be lost if the pull moved files under us: tracked edits only, untracked notes stay put
    dirty=$(git status --porcelain --untracked-files=no)
    branch=$(git symbolic-ref --short -q HEAD || echo "")
    echo "Updating r9700-stack (branch ${branch:-detached HEAD}) ..."
    if [ -n "$dirty" ]; then
      echo "You have edits to tracked files:"
      echo "$dirty" | sed 's/^/  /'
      echo "Nothing was pulled. Commit or stash them first (git status lists the same files), or run"
      echo "\"git pull --ff-only\" yourself and then ./update.sh."
      exit 1
    fi
    if ! git pull --ff-only; then
      echo
      echo "git pull did not succeed (the reason is above): nothing was updated. This repository's origin is"
      echo "the fork you push to; the bkvargyas/upstream remote is fetch-only by design."
      exit 1
    fi
  fi
  # the running plugin is not read from this folder: it is read from the build repo serve.sh mounts, so
  # say what that one is on before setup.py compares them
  repo=${REPO:-$HOME/r9700-build/repo}
  if [ -d "$repo/.git" ]; then
    echo "The build repo serve.sh mounts is $repo at $(git -C "$repo" rev-parse --short HEAD 2>/dev/null || echo '?')."
    if [ -n "$(git -C "$repo" status --porcelain --untracked-files=no)" ]; then
      echo "It carries its own edits, so setup.py will not move it -- update it by hand if you want them gone."
    fi
  fi
  exec sh ./setup.sh --update "$@"
}
main "$@"
