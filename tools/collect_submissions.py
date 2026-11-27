"""按每行一个的 GitHub 学号分支链接，下载供助教批改的作业。"""

import argparse
import csv
from pathlib import Path
import re
import subprocess
import sys
import tempfile


BRANCH_URL = re.compile(
    r"https://github\.com/([\w.-]+)/([\w.-]+)/tree/([\w-]+)/?"
)


def read_links(path):
    submissions = {}
    for line, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        url = raw.strip()
        if not url:
            continue
        match = BRANCH_URL.fullmatch(url)
        if not match:
            raise ValueError(f"第 {line} 行需要学号分支链接，例如 https://github.com/用户名/仓库/tree/20260001")
        owner, repo, student = match.groups()
        if student in ("main", "master", "submissions") or "." in (owner, repo) or ".." in (owner, repo):
            raise ValueError(f"第 {line} 行需要学生自己 fork 中的学号分支：{url}")
        remote = f"https://github.com/{owner}/{repo}.git"
        key = student.casefold()
        if key in submissions and submissions[key] != (student, remote):
            raise ValueError(f"第 {line} 行学号 {student} 对应多个提交，请核对清单。")
        submissions[key] = (student, remote)
    if not submissions:
        raise ValueError("清单为空，请每行填写一个学号分支链接。")
    return submissions.values()


def snapshot(student, remote, destination):
    folder = destination / student
    if not folder.exists():
        with tempfile.TemporaryDirectory(dir=str(destination)) as staging:
            checkout = Path(staging) / "checkout"
            subprocess.run(
                ["git", "clone", "--quiet", "--depth=1", "--single-branch", "--branch", student,
                 "--", remote, str(checkout)], check=True,
            )
            head = git_value(checkout, "symbolic-ref", "--short", "HEAD")
            if head != student:
                raise ValueError(f"{student} 的链接没有指向同名分支。")
            checkout.rename(folder)
    if not (folder / ".git").is_dir() or git_value(folder, "config", "--local", "--get", "remote.origin.url") != remote:
        raise ValueError(f"已有目录 {folder} 与提交链接不符，请使用新的收取目录。")
    # Read the original downloaded commit even if the grader has edited the working tree.
    return git_value(folder, "rev-parse", "--verify", f"refs/remotes/origin/{student}^{{commit}}")


def git_value(folder, *args):
    return subprocess.check_output(["git", "-C", str(folder), *args], encoding="utf-8").strip()


def collect(links, destination):
    submissions = read_links(links)
    destination.mkdir(parents=True, exist_ok=True)
    rows = []
    failed = []
    for student, remote in submissions:
        try:
            sha = snapshot(student, remote, destination)
            rows.append((student, f"{remote[:-4]}/tree/{student}", sha))
        except (subprocess.CalledProcessError, ValueError, OSError) as error:
            failed.append(student)
            print(f"{student} 收取失败：{error}", file=sys.stderr)
    with tempfile.NamedTemporaryFile(mode="w", dir=str(destination), encoding="utf-8",
                                     newline="", delete=False) as output:
        writer = csv.writer(output, delimiter="\t")
        writer.writerow(["学号", "分支链接", "提交号"])
        writer.writerows(rows)
    Path(output.name).replace(destination / "submissions.tsv")
    print(f"{len(rows)} 份作业已收取到 {destination}。")
    if failed:
        raise SystemExit("请核对以下学号的链接后重新运行：" + "、".join(failed))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("links", type=Path, help="链接清单，每行一份作业")
    parser.add_argument("directory", type=Path, help="本次作业的保存目录")
    args = parser.parse_args()
    try:
        collect(args.links, args.directory)
    except (ValueError, OSError) as error:
        parser.exit(1, f"{error}\n")
