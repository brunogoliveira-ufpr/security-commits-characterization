import json
from pydriller import RepositoryMining
path = './'


def summaryModifications(modifications):

    added = 0
    removed = 0
    changed_methods = 0
    diff_methods = 0
    complexity = 0
    for m in modifications:

        #filename = m.filename
        added += m.added
        removed += m.removed
        changed_methods += len(m.changed_methods)
        diff_methods += len(m.methods) - len(m.methods_before)
        if m.complexity is not None:
            complexity += m.complexity

    if complexity != 0:
        complexity = complexity / len(modifications)

    return [len(modifications), added, removed, added + removed,
            diff_methods, changed_methods, complexity] #, filename]


def collectPyDrillerMetrics(commitHash, repoPath):
    rm = RepositoryMining(repoPath, only_commits=[commitHash],only_no_merge=True)
    resp = []
    for commit in rm.traverse_commits():
        #resp.append(commit.dmm_unit_size)
        #resp.append(commit.dmm_unit_complexity)
        resp.append(commit.hash)
        resp.extend(summaryModifications(commit.modifications))
        #resp.extend(findKeywords(commit.msg))

    return [str(e) for e in resp]


def main():
        with open('commits_not.txt') as f:
                lines = [ line.strip() for line in f ]

                for line1 in lines:
                        results = ('results_not_pydriller.csv')
                        file2 = open(results, 'a')
                        file2.write(str(collectPyDrillerMetrics(line1,path))+"\n")

if __name__ == "__main__":
        main()