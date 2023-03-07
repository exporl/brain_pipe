The contribution guide for BIDS preprocessing 
=============================================
Thank you for investing your time in contributing to our project!

Following these guidelines helps to communicate that you respect the time of the 
developers managing and developing this open source project. In return, they should 
reciprocate that respect in addressing your issue, assessing changes, and helping you 
finalize your pull requests.

In this guide you will get an overview of the contribution workflow from opening an 
issue, creating a PR, reviewing, and merging the PR.

Please, don't use the issue tracker for general programming support questions, you can
use sites like [Stack Overflow](https://stackoverflow.com) for that.


Motivation and scope
--------------------

The main goal of this repository is to provide a python3 package to efficiently 
preprocess BIDS datasets. The 'efficiently' in this context means that the code should
be able to preprocess as much data as possible in a given timespace, maximally utilizing
the available system resources. The main application of this code is to preprocess large
datasets for machine learning applications

Using the issue tracker
-----------------------

The issue tracker is the preferred channel for [bug reports](#bugs),
[features requests](#features) and [submitting pull
requests](#pull-requests), but please respect the following restrictions:

* Please **do not** use the issue tracker for personal support requests (use
  [Stack Overflow](http://stackoverflow.com)).

* Please **do not** derail or troll issues. Keep the discussion on topic and
  respect the opinions of others.

<a name="code-quality"></a>
Code quality
------------
* All code should be formatted with [black](https://www.github.com/psf/black).
* PEP compliance and documentation is checked with [flake8](https://www.github.com/PyCQA/flake8) and [flake8-docstrings](https://github.com/pycqa/flake8-docstrings).
* Test code is provided in the [tests](./tests) folder and are in the [unittest style](https://docs.python.org/3/library/unittest.html)
* When adding new code, please add tests for it as well.



<a name="bugs"></a>
Bug reports
-----------

A bug is a _demonstrable problem_ that is caused by the code in the repository.
Good bug reports are extremely helpful - thank you!

Guidelines for bug reports:

1. **Use the GitHub issue search** &mdash; check if the issue has already been
   reported.

2. **Check if the issue has been fixed** &mdash; try to reproduce it using the
   latest `master` or development branch in the repository.

3. **Isolate the problem** &mdash; create a [reduced test
   case](http://css-tricks.com/reduced-test-cases/).

A good bug report shouldn't leave others needing to chase you up for more
information. Please try to be as detailed as possible in your report. What is
your environment? What steps will reproduce the issue? What browser(s) and OS
experience the problem? What would you expect to be the outcome? All these
details will help people to fix any potential bugs.

Example:

> Short and descriptive example bug report title
>
> A summary of the issue and the browser/OS environment in which it occurs. If
> suitable, include the steps required to reproduce the bug.
>
> 1. This is the first step
> 2. This is the second step
> 3. Further steps, etc.
>
> `<url>` - a link to the reduced test case
>
> Any other information you want to share that is relevant to the issue being
> reported. This might include the lines of code that you have identified as
> causing the bug, and potential solutions (and your opinions on their
> merits).


<a name="features"></a>
Feature requests
----------------

Feature requests are welcome. But take a moment to find out whether your idea
fits with the scope and aims of the project. It's up to *you* to make a strong
case to convince the project's developers of the merits of this feature. Please
provide as much detail and context as possible.


<a name="pull-requests"></a>
Pull requests
-------------
Good pull requests - patches, improvements, new features - are a fantastic
help. They should remain focused in scope and avoid containing unrelated
commits.

**Please ask first** before embarking on any significant pull request (e.g.
implementing features, refactoring code, porting to a different language),
otherwise you risk spending a lot of time working on something that the
project's developers might not want to merge into the project.

Please adhere to the coding conventions used throughout a project (indentation,
accurate comments, etc.) and any other requirements (see also [Code quality](#code-quality)).

Follow this process if you'd like your work considered for inclusion in the
project:

1. [Fork](http://help.github.com/fork-a-repo/) the project, clone your fork,
   and configure the remotes:

   ```bash
   # Clone your fork of the repo into the current directory
   git clone https://github.com/<your-username>/<repo-name>
   # Navigate to the newly cloned directory
   cd <repo-name>
   # Assign the original repo to a remote called "upstream"
   git remote add upstream https://github.com/<upstream-owner>/<repo-name>
   ```

2. If you cloned a while ago, get the latest changes from upstream:

   ```bash
   git checkout <dev-branch>
   git pull upstream <dev-branch>
   ```

3. Create a new topic branch (off the main project development branch) to
   contain your feature, change, or fix:

   ```bash
   git checkout -b <topic-branch-name>
   ```

4. Commit your changes in logical chunks. Please adhere to these [git commit
   message guidelines](http://tbaggery.com/2008/04/19/a-note-about-git-commit-messages.html)
   or your code is unlikely be merged into the main project. Use Git's
   [interactive rebase](https://help.github.com/articles/interactive-rebase)
   feature to tidy up your commits before making them public.

5. Locally merge (or rebase) the upstream development branch into your topic branch:

   ```bash
   git pull [--rebase] upstream <dev-branch>
   ```

6. Push your topic branch up to your fork:

   ```bash
   git push origin <topic-branch-name>
   ```

7. [Open a Pull Request](https://help.github.com/articles/using-pull-requests/)
    with a clear title and description.

**IMPORTANT**: By submitting a patch, you agree to allow the project owner to
license your work under the same license as that used by the project.


(This document is based on [@nayafia's example](https://github.com/nayafia/contributing-template/blob/HEAD/CONTRIBUTING-template.md) and the [contributing guide of mrtfpy](https://github.com/powerfulbean/mTRFpy/blob/master/CONTRIBUTING.md))