# Characterizing Security-related Commits of JavaScript Engines

This tutorial is a guide to replicate the results from the article. 

## Projects	
Four JavaScript engines were chosen for the assesment, they are all GitHub open-source projects: v8, ChakraCore, JavaScriptCore and Hermes. 
To start the process is necessary to download the projects.

git clone https://github.com/v8/v8

git clone https://github.com/chakra-core/ChakraCore

git clone https://github.com/WebKit/WebKit

git clone https://github.com/facebook/hermes

## Commits
We collected the commits hashes from all packages utilizing the git comnmand-line utility inside each project folder.

git log --pretty=format:%H > project_hashes.txt

Since JavaScriptCore is inside the WebKit package, we collected only commits related to the JavaScriptCore folder.

git log --pretty=format:%H -- Source/JavaScriptCore > JavaScriptCore_hashes.txt

## Tools
It was utilized two tools for extracting the metrics. PyDriller library and Understand.
To install the PyDriller: pip3 install pydriller

To download and install Understand, it is required to register at https://www.scitools.com/.

## Methodology

### RQ1

We used the following script to extract messages from each project: pygithub_getmessages.py.

Then, we utilized scikit-learn library to identify security-related commits by their messages: classifier.py. We randomly selected other commits for the evaluation, same amount of security-related commits for each engine.

Then we extracted software metrics with PyDriller using the script: py_tests.py and Understand using the bash script: und_extraction.sh. We utilized the processing.py Python script to calculate the statistics test and effect size from each metric.

| Metric                      | What is |
|-----------------------------|-------------|
| lines_added                 | Number of lines added in a commit or change. |
| lines_removed               | Number of lines removed in a commit or change. |
| lines_added+removed         | Total lines changed (added + removed). |
| diff_methods                | Methods with any differences between versions. |
| changed_methods             | Number of methods that have been changed. |
| AltAvgLineBlank             | Average number of blank lines per file (alternative tool). |
| AltAvgLineCode              | Average number of code lines per file (alternative tool). |
| AltAvgLineComment           | Average number of comment lines per file (alternative tool). |
| AltCountLineBlank           | Total blank lines (alternative tool). |
| AltCountLineComment         | Total comment lines (alternative tool). |
| AvgCyclomatic               | Average cyclomatic complexity across methods. |
| AvgCyclomaticModified       | Average modified cyclomatic complexity. |
| AvgCyclomaticStrict         | Average strict cyclomatic complexity. |
| AvgEssential                | Average essential complexity (structuredness). |
| AvgLine                     | Average number of lines per method or file. |
| AvgLineBlank                | Average blank lines per method or file. |
| AvgLineCode                 | Average code lines per method or file. |
| AvgLineComment              | Average comment lines per method or file. |
| CountDeclClass              | Number of class declarations. |
| CountLine                   | Total lines of code. |
| CountLineBlank              | Number of blank lines. |
| CountLineComment            | Number of comment lines. |
| CountLineInactive           | Number of lines excluded by preprocessor directives. |
| CountSemicolon              | Total number of semicolons (can approximate statement count). |
| CountStmt                   | Total number of statements. |
| CountStmtDecl               | Number of declaration statements. |
| CountStmtEmpty              | Number of empty statements. |
| CountStmtExe                | Number of executable statements. |
| MaxCyclomatic               | Maximum cyclomatic complexity of any method. |
| MaxCyclomaticModified       | Maximum modified cyclomatic complexity. |
| MaxEssential                | Maximum essential complexity of any method. |
| RatioCommentToCode          | Ratio of comment lines to code lines. |
| SumCyclomatic               | Total cyclomatic complexity across all methods. |
| SumCyclomaticModified       | Total modified cyclomatic complexity. |
| AltCountLineCode            | Total number of code lines (alternative tool). |
| CountLineCodeDecl           | Number of lines with code declarations. |
| CountDeclFunction           | Number of function declarations. |
| CountLinePreprocessor       | Number of preprocessor directive lines. |
| CountLineCode               | Number of code lines. |
| CountLineCodeExe            | Number of executable code lines. |
| MaxCyclomaticStrict         | Maximum strict cyclomatic complexity. |
| SumCyclomaticStrict         | Sum of strict cyclomatic complexity across methods. |
| SumEssential                | Sum of essential complexity across methods. |
| MaxNesting                  | Maximum nesting depth in control structures. |


### RQ2

We verified what files are being modified by security-related commits and others using the data provided by PyDriller and Understand. We counted the files that resulted on the top files modified. We analyzed the file and related to the module manually by searching for keywords, observing the folder path and analyzing the functions and classes.

### RQ3

We randomly selected 5% from the security-related commits and classified the type of vulnerability. We increased the JavaScriptCore and Hermes number to 50, to have enough data for the classification. The type of vulnerability identification was done by seaching for keywords on the title and messages, inspecting external references (as CVEs) and finally interpreted the messages for classification.
