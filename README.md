Characterizing Security-related Commits of JavaScript Engines

This tutorial is a guide to replicate the results from the article. 

Projects
Commits
Tools
Methodology

- Projects	
The four JavaScript engines chosen for the assesment are GitHub open-source projects: v8, ChakraCore, JavaScriptCore and Hermes. 
To start the process is necessary to download the projects.

git clone https://github.com/v8/v8
git clone https://github.com/chakra-core/ChakraCore
git clone https://github.com/WebKit/WebKit
git clone https://github.com/facebook/hermes

- Commits
We collected the commits hashes from all packages utilizing the git comnmand-line utility inside each project folder.

git log --pretty=format:%H > $project_hashes.txt

Since JavaScriptCore is inside the WebKit package, we collected only commits related to the JavaScriptCore folder.

git log --pretty=format:%H -- Source/JavaScriptCore > JavaScriptCore_hashes.txt

- Tools
It was utilized two tools for extracting the metrics. PyDriller library and Understand.
To install the PyDriller: pip3 install pydriller

To download and install Understand, it is required to register at https://www.scitools.com/.

- Methodology

RQ1

We utilized the following script for extracting the messages from each project: pygithub_getmessages.py
Then, we utilized scikit-learn library to identify security-related commits by their messages: classifier.py. We randomly selected other commits for the evaluation, same amount of security-related commits for each engine.

Then we extracted software metrics with PyDriller using the script: py_tests.py and Understand using the bash script: und_extraction.sh. We utilized the processing.py Python script to calculate the statistics test and effect size from each metric.

RQ2

We verified what files are being modified by security-related commits and others using the data provided by PyDriller and Understand. We counted the files that resulted on the top files modified. We analyzed the file and related to the module manually by searching for keywords, observing the folder path and analyzing the functions and classes.

RQ3

We randomly selected 5% from the security-related commits and classified the type of vulnerability. We increased the JavaScriptCore and Hermes number to 50, to have enough data for the classification. The type of vulnerability identification was done by seaching for keywords on the title and messages, inspecting external references (as CVEs) and finally interpreted the messages for classification.
