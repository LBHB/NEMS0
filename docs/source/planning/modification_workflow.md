From Brad, suggested procedure for modifying master branch:

```
# Create a new "feature" or "bugfix" branch
git branch -b bburan/signal-slicing-dicing
# Add the files relevant to the branch and commit them
git add <files I changed relevant to the commit>
git commit
# Now, run the unit-tests to make sure everything passes!
cd <path to NEMS>/tests
pytest
# Hide all other pending changes. Probably not necessary.
git stash
# Switch back to master and pull in all pending changes
git checkout master
git pull
# Switch back to our branch and rebase off of master
git checkout bburan/signal-slicing-dicing
git rebase master
# Now, push it to GitHub
git push origin bburan/signal-slicing-dicing
# Now, go to GitHub and issue a pull request
<not a command-line option, I think>
# Now, if you want to continue working as if the branch was already pulled into master, create your own personal branch (call it whatever you like, mine is called bburan/rdt). Use the -b if the branch doesn't exist yet.
git checkout -b bburan/master 
git merge master
git merge bburan/signal-slicing-dicing
git merge <other pending pull requests you want to use>
# There's also other commands like git cherry-pick, but I'm not an expert on these.
```
