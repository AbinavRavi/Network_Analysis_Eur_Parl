# Network_Analysis_Eur_Parl
A project using community detection to find the Hidden Agenda in MP of european parliament. The data is taken from 1999-2012 for which the English speeches are available. A project for Data Innovation Lab by Niklas Schmidt and Abinav Ravi

#Data:
To get the data run the SQl queries files in Data and join them.

#Code
1.First run the Coherence topic model to extract the topics in the required format. The output of this file is a .csv file which contains the names of MEP with a list of topics that they have been associated with in decreasing order of probability

2. With this data run the Network_Model_HiCode.py file in Code to get a list of 2 MEP who has interacted more in the Hidden community or in the second layer of relevance. Implementation Done according to Hidden Community Detection by He.et.al. The paper can be found in https://arxiv.org/abs/1702.07462.

3. With the same data run network_model_HiAgDe which detects outliers in the community after a round of community detection.
