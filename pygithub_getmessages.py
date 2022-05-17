# -*- coding: utf-8 -*-
 
from github import Github
 
# Generate a token at https://github.com/settings/tokens
token = ''
results = ('v8_messages_pygithub.csv')
 
g = Github(token)
 
repo_name = 'v8/v8'
repo = g.get_repo(repo_name)
 
with open('all_commits.txt') as f:
	lines = [line.rstrip() for line in f]
	for line in lines:
		commit = repo.get_commit(line)
		#commit_title = commit.commit.title.replace(',', '').encode('utf-8').strip()
		commit_msg = commit.commit.message.replace(',', '').encode('utf-8').strip()
		file = open(results, 'a')
		file.write(line+','+str(commit_msg).replace('\n\n','').replace('\r','').replace('\n','')+'\n')
		#print(commit.commit.message)