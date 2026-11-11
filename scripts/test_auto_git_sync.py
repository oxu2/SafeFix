"""CPU-only integration tests using isolated local Git repositories."""
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

spec = importlib.util.spec_from_file_location('auto_git_sync', Path(__file__).with_name('auto_git_sync.py'))
syncer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(syncer)


class AutoGitSyncTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root = self.base / 'work'
        self.root.mkdir()
        self.remote = self.base / 'remote.git'
        self.git('init', '--initial-branch=master')
        self.git('config', 'user.name', 'Offline Test')
        self.git('config', 'user.email', 'offline@example.invalid')
        self.git('config', 'commit.gpgsign', 'false')
        subprocess.run(['git', 'init', '--bare', str(self.remote)], check=True, capture_output=True)
        self.git('remote', 'add', 'origin', str(self.remote))
        (self.root / 'file.txt').write_text('base\n')
        self.git('add', '.')
        self.git('commit', '-m', 'base')

    def git(self, *args, check=True):
        return subprocess.run(['git', *args], cwd=self.root, check=check, text=True, capture_output=True)

    def remote_head(self):
        return self.git('ls-remote', 'origin', 'refs/heads/master').stdout.split()[0]

    def changed_file(self):
        (self.root / 'file.txt').write_text('new content\n')
        return syncer.changed(self.root)

    def test_missing_branch_is_created_without_merging_unrelated_default(self):
        # A remote main may exist with entirely unrelated history.
        other = self.base / 'other'
        subprocess.run(['git', 'init', '--initial-branch=main', str(other)], check=True, capture_output=True)
        for key, value in [('user.name', 'Other'), ('user.email', 'other@example.invalid'), ('commit.gpgsign', 'false')]:
            subprocess.run(['git', '-C', str(other), 'config', key, value], check=True)
        subprocess.run(['git', '-C', str(other), 'commit', '--allow-empty', '-m', 'unrelated'], check=True, capture_output=True)
        subprocess.run(['git', '-C', str(other), 'push', str(self.remote), 'main'], check=True, capture_output=True)
        self.assertTrue(syncer.sync(self.root, 'master', self.changed_file()))
        self.assertEqual(self.remote_head(), self.git('rev-parse', 'HEAD').stdout.strip())
        self.assertEqual(self.git('rev-list', '--count', 'HEAD').stdout.strip(), '2')
        self.assertEqual(self.git('status', '--porcelain').stdout, '')

    def test_failed_push_clean_worktree_retries_via_main(self):
        self.git('push', 'origin', 'master')
        hook = self.remote / 'hooks' / 'pre-receive'
        hook.write_text('#!/bin/sh\nexit 1\n')
        hook.chmod(0o755)
        self.assertFalse(syncer.sync(self.root, 'master', self.changed_file()))
        local_head = self.git('rev-parse', 'HEAD').stdout.strip()
        self.assertNotEqual(self.remote_head(), local_head)
        self.assertEqual(syncer.changed(self.root), [])
        hook.unlink()
        with mock.patch.object(sys, 'argv', ['auto_git_sync.py', '--repo', str(self.root), '--branch', 'master']):
            syncer.main()
        self.assertEqual(self.remote_head(), local_head)
        self.assertEqual(self.git('rev-list', '--count', 'HEAD').stdout.strip(), '2')

    def test_existing_conflict_is_not_staged_committed_or_aborted(self):
        self.git('checkout', '-b', 'other')
        (self.root / 'file.txt').write_text('other\n')
        self.git('commit', '-am', 'other')
        self.git('checkout', 'master')
        (self.root / 'file.txt').write_text('master\n')
        self.git('commit', '-am', 'master')
        self.assertNotEqual(self.git('merge', 'other', check=False).returncode, 0)
        before = self.git('status', '--porcelain').stdout
        head = self.git('rev-parse', 'HEAD').stdout
        content = (self.root / 'file.txt').read_bytes()
        self.assertFalse(syncer.sync(self.root, 'master', syncer.changed(self.root)))
        self.assertEqual(self.git('status', '--porcelain').stdout, before)
        self.assertEqual(self.git('rev-parse', 'HEAD').stdout, head)
        self.assertEqual((self.root / 'file.txt').read_bytes(), content)
        self.assertTrue((self.root / '.git' / 'MERGE_HEAD').exists())

    def test_remote_error_does_not_attempt_pull_or_push(self):
        self.git('remote', 'set-url', 'origin', str(self.base / 'missing.git'))
        original = syncer.run
        commands = []

        def capture(args, *a, **kw):
            commands.append(args)
            return original(args, *a, **kw)

        with mock.patch.object(syncer, 'run', capture):
            self.assertFalse(syncer.sync(self.root, 'master', []))
        self.assertFalse(any(args[1] in ('pull', 'push') for args in commands))

    def test_pull_error_does_not_fall_through_to_push(self):
        self.git('push', 'origin', 'master')
        self.git('commit', '--allow-empty', '-m', 'pending')
        original = syncer.run
        commands = []

        def fail_pull(args, *a, **kw):
            commands.append(args)
            if args[1] == 'pull':
                return subprocess.CompletedProcess(args, 1, '', 'simulated fetch failure')
            return original(args, *a, **kw)

        with mock.patch.object(syncer, 'run', fail_pull):
            self.assertFalse(syncer.sync(self.root, 'master', []))
        self.assertFalse(any(args[1] == 'push' for args in commands))

    def test_no_push_commits_without_accessing_remote(self):
        self.git('remote', 'remove', 'origin')
        self.assertTrue(syncer.sync(self.root, 'master', self.changed_file(), no_push=True))
        self.assertEqual(self.git('rev-list', '--count', 'HEAD').stdout.strip(), '2')
        with mock.patch.object(syncer, 'run', side_effect=AssertionError('no Git operation expected')):
            self.assertTrue(syncer.sync(self.root, 'master', [], no_push=True))


if __name__ == '__main__':
    unittest.main()
