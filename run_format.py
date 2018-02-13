
# Copyright (c) 2017 Isuru Fernando

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

from contextlib import contextmanager
import os
import tornado.httpserver
import tornado.ioloop
import tornado.web
import requests
import tempfile
import shutil
import subprocess

from git import GitCommandError, Repo, Actor
import github
from fnmatch import fnmatch

@contextmanager
def tmp_directory():
    tmp_dir = tempfile.mkdtemp('_clang_format')
    yield tmp_dir
    shutil.rmtree(tmp_dir)

def get_github_token():
    tok = os.environ.get('GH_TOKEN')
    if not tok:
        raise RuntimeError("GH_TOKEN not available")
    return tok

def get_all_files(tmp_dir):
    l = []
    for root, directories, filenames in os.walk(tmp_dir):
        d = os.path.relpath(root, tmp_dir)
        if d == ".git" or d.startswith(".git/"):
            continue
        for filename in filenames: 
            l.append(os.path.relpath(os.path.join(root, filename), tmp_dir)) 
    return l
    
def run_clang_format(pr_id, gh_repo, gh):

    with tmp_directory() as tmp_dir:
        repo = Repo.clone_from(gh_repo.clone_url, tmp_dir)

        # Retrieve the PR refs.
        repo.remotes.origin.fetch([
            'pull/{pr}/head:pull/{pr}/head'.format(pr=pr_id)
        ])
        ref_head = repo.refs['pull/{pr}/head'.format(pr=pr_id)]
        ref_head.checkout(force=True)
        files = get_all_files(tmp_dir)
        includes = []
        excludes = []
        version = ""
        clang_format_file = '{}/.clang-format'.format(tmp_dir)
        if not os.path.isfile(clang_format_file):
            print("clang-format file not found")
        with open(clang_format_file) as f:
            content = f.readlines()
        content = [x.strip() for x in content]
        for line in content:
            if line.startswith("# version: "):
                version = line[len("# version: "):]
            if line.startswith("# include: "):
                includes.append(line[len("# include: "):])
            if line.startswith("# exclude: "):
                excludes.append(line[len("# exclude: "):])
        print(version, includes, excludes)
        if not version in ["3.3", "3.4", "3.5", "3.6", "3.7", "3.8", "3.9"]:
            print("clang-format version not supported")
        final_file_list = []
        for f in files:
            if any(fnmatch(f, include) for include in includes) and not(any(fnmatch(f, exclude) for exclude in excludes)):
                final_file_list.append(f)
        print(final_file_list)
        for f in final_file_list:
            subprocess.check_call(["clang-format-{}".format(version), "-i", f], cwd=tmp_dir)
        gh_botname = gh.get_user().login
        gh_username = gh_repo.full_name.split("/")[0]
        gh_reponame = gh_repo.name
        actor = Actor(gh.get_user().name, "{}@users.noreply.github.com".format(gh_botname))
        print(subprocess.check_output(["git", "diff"], cwd=tmp_dir))
        if len(repo.index.diff(None)) > 0:
            repo.git.add(u=True)
            commit = repo.index.commit("Format using clang-format-{}".format(version), author=actor, committer=actor)
            dest_url = "https://{}@github.com/{}/{}".format(get_github_token(), gh_botname, gh_reponame)
            remote = repo.create_remote(gh_botname, url=dest_url)
            remote.push(refspec='{}:{}'.format("HEAD", "format-{}-pr-{}".format(gh_username, pr_id)), force=True)
            commit_url = "https://github.com/{}/{}/commit/{}".format(gh_botname, gh_reponame, commit.hexsha)

            msg = """
Hi,
I've run clang-format and found that the code needs formatting.
Here's a commit that fixes this. {}
To use the commit you can do
    curl -o format.diff {}.diff
    git apply format.diff
"""
            msg = msg.format(commit_url, commit_url)
            issue = gh_repo.get_issue(pr_id)
            issue.create_comment(msg)

        return


class MainHandler(tornado.web.RequestHandler):
    def post(self):
        headers = self.request.headers
        event = headers.get('X-GitHub-Event', None)

        if event == 'ping':
            self.write('pong')
        elif event == 'pull_request':
            body = tornado.escape.json_decode(self.request.body)
            action = body['action']
            title = body['pull_request']['title']
            pr = int(body['pull_request']['number'])
            repo_slug = body['repository']['full_name']
            if action == "opened" or action == "synchronize":
                print("Linting PR {} from {}".format(pr, repo_slug))
                gh = github.Github(get_github_token())
                gh_repo = gh.get_repo(repo_slug)
                run_clang_format(pr, gh_repo, gh)

def main():
    application = tornado.web.Application([
        (r"/", MainHandler),
    ])
    http_server = tornado.httpserver.HTTPServer(application)
    port = int(os.environ.get("PORT", 5000))
    http_server.listen(port)
    tornado.ioloop.IOLoop.instance().start()

if __name__ == "__main__":
    main()
