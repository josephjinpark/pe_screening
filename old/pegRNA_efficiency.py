#!/home/jinmanlab/anaconda3/bin/python3.6


import pandas as pd
import Bio as bio
from Bio import SeqIO
import glob, sys

sInputDir  = '/data/scripts/pe_screening/input'
sOutputDir = '/data/scripts/pe_screening/output'

#generates list of fastq files to analyze
sFastqDir     = '%s/TestData/Fig4f' % sInputDir
list_sInFiles = glob.glob('%s/*.fastq' % sFastqDir)

#reads the fastq files into a dictionary with the file names as keys
dict_sSeq = {}

for sInFile in list_sInFiles:

    print('Parsing %s' % sInFile)

    sParseObject        = list(SeqIO.parse(sInFile,"fastq"))
    dict_sSeq[sInFile]  = [str(sParseObject[k].seq) for k in range(len(sParseObject))]
#loop END: sInFile

#the referenced sequence to be searched for is entered into the following dictionary with
#an appropriate key
dict_sScaffSeq = {'HEK3':'CAGAGGACCGACTCGGTCCCACTTTTTCAAGTTGATAACGGACTAGCCTTATTTTAACTTGCTATTTCTAGCTCTAAAACTCACGTGCTCAGTCTGGGCCGGTG',
                  'EMX1':'ATCACGCACCGACTCGGTGCCACTTTTTCAAGTTGATAACGGACTAGCCTTATTTTAACTTGCTATTTCTAGCTCTAAAACTTCTTCTTCTGCTCGGACTCGGTG',
                  'FANCF':'TTTCCGCACCGACTCGGTGCCACTTTTTCAAGTTGATAACGGACTAGCCTTATTTTAACTTGCTATTTCTAGCTCTAAAACGGTGCTGCAGAAGGGATTCCGGTG',
                  'RNF2':'TCGTTGCACCGACTCGGTGCCACTTTTTCAAGTTGATAACGGACTAGCCTTATTTTAACTTGCTATTTCTAGCTCTAAAACCAGGTAATGACTAAGATGACGGTG',
                 }


#matches and counts iterative slices of the reference string to the appropriate fastq files
#reference key must be contained in the name of the fastq file
#generated values represent cumulative counts for a minimum degree of sgRNA integration
#i.e. a given value x means x reads contain y or more bases of the scaffold

dict_sResults = dict.fromkeys(list_sInFiles[:1])

for sInFile in dict_sSeq:
    print('sFastq Seq Cnt', len(dict_sSeq[sInFile]))

    for sScaffoldSeq in dict_sScaffSeq:

        list_sResults = []

        if not sScaffoldSeq in str(sInFile): continue

        for j in range(len(dict_sScaffSeq[sScaffoldSeq])):

            sSubSeq  = dict_sScaffSeq[sScaffoldSeq][0:(j+1)]
            nCnt     = 0

            for sSeq in dict_sSeq[sInFile]:
                if sSubSeq in sSeq: nCnt += 1
            # loop END: sSeq
            list_sResults.append(nCnt)
        #loop END: j
        dict_sResults[sInFile] = list_sResults
    #loop END: sScaffoldSeq
#loop END: key

#writes the results into a dataframe indexed from 1
resultdf = pd.DataFrame.from_dict(dict_sResults)
resultdf = resultdf.reindex(sorted(resultdf.columns), axis=1)
resultdf.index = range(1,len(resultdf)+1)
resultdf.to_excel('cumulativecounts.xlsx')

#converts the cumulative count values into specific counts
#i.e. a given value x means x reads contain exactly y bases of the scaffold
resultdf2=resultdf.copy()
for entry in resultdf:
    for i in range(1,len(resultdf[entry])+1):
        try: resultdf2[entry][i] = resultdf[entry][i]-resultdf[entry][i+1]
        except: resultdf2[entry][i] = resultdf[entry][i]
resultdf2.to_excel('specificcounts.xlsx')


#converts the specific counts values into frequencies
resultdf3=resultdf2.copy()
for entry in resultdf3:
    resultdf3[entry]=resultdf2[entry].div(resultdf[entry][1])*100

resultdf3.to_excel('specificfrequencies.xlsx')