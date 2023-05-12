#!/home/hkim/anaconda3/bin/python3

import os, sys, pickle, time, subprocess, json, re, regex, random
from Bio.SeqUtils import MeltingTemp as mt
from Bio.SeqUtils import GC as gc
from Bio.Seq import Seq


sTIME_STAMP = '%s'          % (time.ctime().replace(' ', '-').replace(':', '_'))

class cFeatures:
    def __init__(self):

        cFeatures.sTarSeq    = ''
        cFeatures.sAltSeq    = ''
        cFeatures.sGuideSeq  = ''
        cFeatures.list_sSeqs = []
        cFeatures.fTm1       = 0.0
        cFeatures.fTm2       = 0.0
        cFeatures.fTm3       = 0.0
        cFeatures.fTm4       = 0.0
        cFeatures.fTmD       = 0.0
        cFeatures.fMFE1      = 0.0
        cFeatures.fMFE2      = 0.0
        cFeatures.fMFE3      = 0.0
        cFeatures.fMFE4      = 0.0
        cFeatures.fMFE5      = 0.0
        cFeatures.nGCcnt1    = 0
        cFeatures.nGCcnt2    = 0
        cFeatures.nGCcnt3    = 0
        cFeatures.fGCcont1   = 0.0
        cFeatures.fGCcont2   = 0.0
        cFeatures.fGCcont3   = 0.0
    #def END: __init__

def main():

    ## Constants ##
    dict_sSubType   = {'A':'t', 'T':'a', 'C':'g', 'G':'c'}
    nNickIndex      = 21
    nMutIndex       = 25

    ## Options For Part 1 ##
    list_nPBSLenPt1 = [7, 9, 11, 13, 15, 17]
    list_nRTLenPt1  = [10, 12, 15, 20]

    ## Options For Part 2 ##
    list_nEditPt2   = [i+1 for i in range(9) if i+1 != 5] + [11, 14]
    nPBSLenPt2      = 13
    nRTLenPt2       = 20

    ## Options For Part 3 ##
    list_nEditPt3   = [(1, 2), (1, 5), (1, 10), (2, 3), (2, 5), (2, 10), (5, 6), (5, 10), (10, 11)]
    list_sInserts   = ['a', 'c', 'g', 't', 'ag', 'aggaa', 'aggaatcatg']
    list_nDelSize   = [1, 2, 5, 10]

    nPBSLenPt3      = 13
    nRTLenPt3       = 14 # Can be longer according to insert size

    ## Input ##
    sINPUT_DIR      = '/home/hkim'
    sInputFile      = '%s/Tm_MFE_example.txt' % sINPUT_DIR
    list_sTarSeqs   = load_input_data (sInputFile)
    ############

    ## Output ##
    sOUTPUT_DIR     = '/home/hkim/test'
    os.makedirs(sOUTPUT_DIR, exist_ok=True)
    sOutFile1       = '%s/formodeling.output.pt1.csv' % sOUTPUT_DIR
    sOutFile2       = '%s/formodeling.output.pt2.csv' % sOUTPUT_DIR
    sOutFile3       = '%s/formodeling.output.pt3.csv' % sOUTPUT_DIR
    ############

    dict_cPart1     = {}
    dict_cPart2     = {}
    dict_cPart3     = {}

    for sTarSeq in list_sTarSeqs:

        ## Part 1: +5 G to C conversion ##
        determine_seqs_pt1 (dict_cPart1, sTarSeq, dict_sSubType, nNickIndex, nMutIndex, list_nPBSLenPt1, list_nRTLenPt1)

        ## Part 2: Shifting edit position ##
        determine_seqs_pt2 (dict_cPart2, sTarSeq, dict_sSubType, nNickIndex, list_nEditPt2, nPBSLenPt2, nRTLenPt2)

        ## Part 3: Indel, Subtype Combination ##
        determine_seqs_pt3 (dict_cPart3, sTarSeq, dict_sSubType, nNickIndex, list_sInserts, list_nDelSize, list_nEditPt3, nPBSLenPt3, nRTLenPt3)

    #loop END: sTarSeq

    determine_secondary_structure (dict_cPart1, sOutFile1, nNickIndex, 'Pt1')
    determine_secondary_structure (dict_cPart2, sOutFile2, nNickIndex, 'Pt2')
    determine_secondary_structure (dict_cPart3, sOutFile3, nNickIndex, 'Pt3')

#def END: main


def reverse_complement(sSeq):
    dict_sBases = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'N': 'N', 'U':'U',
                   '.': '.', '*': '*', 'a': 't', 'c': 'g', 'g': 'c', 't': 'a'}
    list_sSeq   = list(sSeq)  # Turns the sequence in to a gigantic list
    list_sSeq   = [dict_sBases[sBase] for sBase in list_sSeq]
    return ''.join(list_sSeq)[::-1]
#def END: reverse_complement


def load_input_data (sInFile):
    InFile       = open(sInFile, 'r')
    list_sOutput = [sReadLine.strip('\n') for sReadLine in InFile]
    InFile.close()
    return list_sOutput
#def END: load_input_data


def determine_seqs_pt1 (dict_cFeat, sTarSeq, sSubType, nNickIndex, nMutIndex , list_nPBSLen, list_nRTLen):

    sRefNuc    = sTarSeq[nMutIndex]
    sAltNuc    = sSubType[sRefNuc]
    sAltSeq    = sTarSeq[:nMutIndex] + sAltNuc + sTarSeq[nMutIndex + len(sAltNuc):]

    for nPBSLen in list_nPBSLen:

        sPBSseq = sTarSeq[nNickIndex - nPBSLen:nNickIndex]

        for nRTLen in list_nRTLen:
            cFeat         = cFeatures()
            cFeat.sTarSeq = sTarSeq
            sRTseqWT      = cFeat.sTarSeq[nNickIndex:nNickIndex + nRTLen]
            sRTseqAlt     = sAltSeq[nNickIndex:nNickIndex + nRTLen]
            sKey          = '%s:%s:%s' % (sPBSseq, sRTseqWT, sRTseqAlt)

            if sKey not in dict_cFeat:
                dict_cFeat[sKey] = ''
            dict_cFeat[sKey] = cFeat
        #loop END: nRTLen
    #loop END: nPBSLen
    return dict_cFeat
#def END: determine_seqs_pt1


def determine_seqs_pt2 (dict_cFeat, sTarSeq, dict_sSubType, nNickIndex, list_nEditPos, nPBSLen, nRTLen):

    sPBSseq    = sTarSeq[nNickIndex - nPBSLen:nNickIndex]
    sRTseqWT   = sTarSeq[nNickIndex:nNickIndex + nRTLen]

    for nEditPos in list_nEditPos:

        cFeat         = cFeatures()
        cFeat.sTarSeq = sTarSeq

        sRTseqAlt     = ''.join([dict_sSubType[sNuc] if (i+1) == nEditPos else sNuc for i, sNuc in enumerate(sRTseqWT)])
        sKey          = '%s:%s:%s' % (sPBSseq, sRTseqWT, sRTseqAlt)

        if sKey not in dict_cFeat:
            dict_cFeat[sKey] = ''
        dict_cFeat[sKey] = cFeat
    #loop END: nEditPos

    return dict_cFeat
#def END: determine_seqs_pt2


def determine_seqs_pt3 (dict_cFeat, sTarSeq, dict_sSubType, nNickIndex, list_sInserts, list_nDelSize, list_nEditPos, nPBSLen, nRTLen):

    sPBSseq    = sTarSeq[nNickIndex - nPBSLen:nNickIndex]

    ## Deletion Set ##
    for nDelSize in list_nDelSize:
        cFeat         = cFeatures()
        cFeat.sTarSeq = sTarSeq
        sRTseqWT      = sTarSeq[nNickIndex:nNickIndex + nRTLen]
        sRTseqAlt     = sTarSeq[nNickIndex + nDelSize:nNickIndex + nRTLen + nDelSize]
        sKey          = '%s:%s:%s' % (sPBSseq, sRTseqWT, sRTseqAlt)

        if sKey not in dict_cFeat:
            dict_cFeat[sKey] = ''
        dict_cFeat[sKey] = cFeat
    #loop END: nDelSize

    ## Insertion Set ##
    for sInsert in list_sInserts:

        cFeat         = cFeatures()
        cFeat.sTarSeq = sTarSeq
        nInSize       = len(sInsert)

        sRTseqWT      = sTarSeq[nNickIndex:nNickIndex + nRTLen + nInSize]
        sRTseqAlt     = sInsert + sTarSeq[nNickIndex:nNickIndex + nRTLen]
        sKey          = '%s:%s:%s' % (sPBSseq, sRTseqWT, sRTseqAlt)

        if sKey not in dict_cFeat:
            dict_cFeat[sKey] = ''
        dict_cFeat[sKey] = cFeat
    #looo END: sInsert

    ## +1 Position: A->T, C, G or C->A, T, G ##
    nRTLen       = 15
    dict_1Pos    = {'A':['c', 'g', 't'], 'C':['a', 't', 'g'],
                    'T':['a', 'c', 'g'], 'G':['a', 'c', 't']}
    sRTseqWT     = sTarSeq[nNickIndex:nNickIndex + nRTLen]
    list_sAltNuc = dict_1Pos[sRTseqWT[0]]

    for sAltNuc in list_sAltNuc:
        cFeat         = cFeatures()
        cFeat.sTarSeq = sTarSeq
        sRTseqAlt     = sAltNuc + sTarSeq[nNickIndex + 1:nNickIndex + nRTLen]
        sKey          = '%s:%s:%s' % (sPBSseq, sRTseqWT, sRTseqAlt)

        if sKey not in dict_cFeat:
            dict_cFeat[sKey] = ''
        dict_cFeat[sKey] = cFeat
    #loop END: sAltNuc

    ## 2 Position Substitution ##
    nRTLen       = 16
    sRTseqWT     = sTarSeq[nNickIndex:nNickIndex + nRTLen]

    for nPos1, nPos2 in list_nEditPos:
        cFeat         = cFeatures()
        cFeat.sTarSeq = sTarSeq

        nPosIndex1    = nPos1 - 1
        nPosIndex2    = nPos2 - 1

        sRefNuc1      = sRTseqWT[nPosIndex1]
        sRefNuc2      = sRTseqWT[nPosIndex2]
        sAltNuc1      = dict_sSubType[sRefNuc1]
        sAltNuc2      = dict_sSubType[sRefNuc2]

        dict_sAltNucs = {nPosIndex1:sAltNuc1, nPosIndex2:sAltNuc2}

        list_sRTSeqAlt = [sNuc if i not in dict_sAltNucs else dict_sAltNucs[i] for i, sNuc in enumerate(sRTseqWT)]
        sRTseqAlt     = ''.join(list_sRTSeqAlt)

        sKey          = '%s:%s:%s' % (sPBSseq, sRTseqWT, sRTseqAlt)
        if sKey not in dict_cFeat:
            dict_cFeat[sKey] = ''
        dict_cFeat[sKey] = cFeat
    #loop END: nEditPos

    return dict_cFeat
#def END: determine_seqs_pt2


def determine_secondary_structure (dict_cFeat, sOutFile, nNickIndex, sPart):

    determine_Tm(dict_cFeat, sPart)
    determine_GC(dict_cFeat)
    determine_MFE(dict_cFeat)

    make_output(dict_cFeat, sOutFile, nNickIndex, sPart)

#def END: determine_secondary_structure



def determine_Tm (dict_cFeat, sPart):

    for sKey in dict_cFeat:

        sPBSseq, sRTseqWT, sRTseqAlt = sKey.split(':')
        cFeat                        = dict_cFeat[sKey]
        #print(sRTseqWT, sRTseqAlt)

        ## Tm1 DNA/RNA mm1 ##
        cFeat.fTm1 = mt.Tm_NN(seq=Seq(reverse_complement(sPBSseq.replace('A','U'))), nn_table=mt.R_DNA_NN1)
        #print('Tm1', cFeat.fTm1)

        ## Tm2 DNA/DNA mm0 ##
        cFeat.fTm2 = mt.Tm_NN(seq=Seq(sRTseqWT.upper()), nn_table=mt.DNA_NN3)

        #print('Tm2', cFeat.fTm2)
        ## Tm3 DNA/DNA mm1 ##
        list_sSeq1 = []
        list_sSeq2 = []
        if sPart != 'Pt3':

            if sPart == 'Pt1': ## 06-30-2020 Temp Adjustment
                sSeq1 = sRTseqWT
                sSeq2 = sRTseqAlt
            elif sPart == 'Pt2':
                sSeq1 = sRTseqAlt
                sSeq2 = sRTseqWT

            for sNT1, sNT2 in zip(sSeq1, sSeq2):
                if sNT1 != sNT2:
                    list_sSeq1.append(sNT1.lower())
                    list_sSeq2.append(sNT2.lower())
                else:
                    list_sSeq1.append(sNT1)
                    list_sSeq2.append(sNT2)
            # loop END: sNT1, sNT2

            sSeq1      = Seq(''.join(list_sSeq1))
            sSeq2      = Seq(reverse_complement(''.join(list_sSeq2))[::-1])
            cFeat.fTm3 = mt.Tm_NN(seq=sSeq1, c_seq=sSeq2, nn_table=mt.DNA_NN3)
            cFeat.fTmD = cFeat.fTm3 - cFeat.fTm2

        else:
            cFeat.fTm3 = ''
            cFeat.fTmD = ''

        #Tm4 - revcom(AAGTcGATCC(RNA version)) + AAGTcGATCC
        cFeat.fTm4 = mt.Tm_NN(seq=Seq(reverse_complement(sRTseqAlt.upper().replace('A','U'))), nn_table=mt.R_DNA_NN1)
    #loop END: sKey

#def END: determine_Tm


def determine_GC (dict_cFeat):
    for sKey in dict_cFeat:
        sPBSseq, sRTseqWT, sRTseqAlt = sKey.upper().split(':')
        cFeat          = dict_cFeat[sKey]
        cFeat.nGCcnt1  = sPBSseq.count('G') + sPBSseq.count('C')
        cFeat.nGCcnt2  = sRTseqAlt.count('G') + sRTseqAlt.count('C')
        cFeat.nGCcnt3  = (sPBSseq + sRTseqAlt).count('G') + (sPBSseq + sRTseqAlt).count('C')
        cFeat.fGCcont1 = gc(sPBSseq)
        cFeat.fGCcont2 = gc(sRTseqAlt)
        cFeat.fGCcont3 = gc(sPBSseq + sRTseqAlt)
    #loop END: sKey
#def END: determine_GC


def determine_MFE (dict_cFeat):

    sScaffoldSeq    = 'GTTTTAGAGCTAGAAATAGCAAGTTAAAATAAGGCTAGTCCGTTATCAACTTGAAAAAGTGGCACCGAGTCGGTGC'
    sPolyTSeq       = 'TTTTTTT'


    for sKey in dict_cFeat:
        sPBSseq, sRTseqWT, sRTseqAlt = sKey.split(':')
        cFeat                        = dict_cFeat[sKey]

        ## Set GuideRNA seq ##
        cFeat.sGuideSeq              = 'G' + cFeat.sTarSeq[5:24]

        #MFE_1 - pegRNA:
        sInputSeq             = cFeat.sGuideSeq + sScaffoldSeq + reverse_complement(sPBSseq + sRTseqAlt) + sPolyTSeq
        sDBSeq, cFeat.fMFE1   = get_dotbracket_and_MFE(sInputSeq, 0.0)

        #MFE_2 - pegRNA without spacer
        sInputSeq             = sScaffoldSeq + reverse_complement(sPBSseq + sRTseqAlt) + sPolyTSeq
        sDBSeq, cFeat.fMFE2   = get_dotbracket_and_MFE(sInputSeq, 0.0)

        #MFE_3 - RT + PBS + PolyT
        sInputSeq             = reverse_complement(sPBSseq + sRTseqAlt) + sPolyTSeq
        sDBSeq, cFeat.fMFE3   = get_dotbracket_and_MFE(sInputSeq, 0.0)

        #MFE_4 - spacer only
        sInputSeq             = cFeat.sGuideSeq
        sDBSeq, cFeat.fMFE4   = get_dotbracket_and_MFE(sInputSeq, 0.0)

        #MFE_5 - Spacer + Scaffold
        sInputSeq             = cFeat.sGuideSeq + sScaffoldSeq
        sDBSeq, cFeat.fMFE5   = get_dotbracket_and_MFE(sInputSeq, 0.0)
        #sDBSeq, fMFE   = get_dotbracket_and_MFE(cRNA.sBarcodeSeq, 0.0)

    #loop END: cRNA
#def END: determine_MFE


def get_dotbracket_and_MFE (sInputSeq,  fMFE):

    #sRNASubopt = '%s/RNAsubopt' % sVIENNA_BIN
    sRNASubopt = 'RNAsubopt'

    sScript      = 'echo "%s" | %s -s -e %d' % (sInputSeq, sRNASubopt, fMFE)
    list_sStdOut = subprocess.Popen(sScript, stdout=subprocess.PIPE, shell=True).stdout

    list_sDotBr  = []

    for i, sReadLine in enumerate(list_sStdOut):
        sReadLine = str(sReadLine, 'UTF-8').strip('\n')

        if i >= 1:
            list_sColumn = sReadLine.split()
            sDotBracket  = list_sColumn[0]

            fMFE      = float(list_sColumn[1])
            list_sDotBr.append([sDotBracket, fMFE])
    #loop END: i, sReadLine

    return list_sDotBr[0]
#def END: get_dotbracket_and_MFE


def make_output (dict_cFeat, sOutFile, nNickIndex, sPart):

    OutFile = open(sOutFile, 'w')
    for sKey in dict_cFeat:
        cFeat                        = dict_cFeat[sKey]
        sPBSseq, sRTseqWT, sRTseqAlt = sKey.split(':')
        sPBS_RTSeq                   = sPBSseq + sRTseqAlt
        s5BufferSeq                  = cFeat.sTarSeq[:nNickIndex-len(sPBSseq)]
        s3BufferSeq                  = cFeat.sTarSeq[nNickIndex + len(sRTseqAlt):]
        sTarSeq_reformat              = 'x' * len(s5BufferSeq) + sPBS_RTSeq + 'x' * len(s3BufferSeq)

        if sPart == 'Pt1':
            list_sOut = [cFeat.sTarSeq, sTarSeq_reformat, len(sPBSseq),
                         len(sRTseqAlt), len(sPBS_RTSeq), cFeat.fTm1, cFeat.fTm2, cFeat.fTm3, cFeat.fTm4, cFeat.fTmD,
                         cFeat.nGCcnt1, cFeat.nGCcnt2, cFeat.nGCcnt3, cFeat.fGCcont1, cFeat.fGCcont2, cFeat.fGCcont3,
                         cFeat.fMFE1, cFeat.fMFE2, cFeat.fMFE3, cFeat.fMFE4, cFeat.fMFE5]
        else:
            list_sOut = [cFeat.sTarSeq, sPBSseq, sRTseqAlt, sPBS_RTSeq, len(sPBSseq),
                         len(sRTseqAlt), len(sPBS_RTSeq), cFeat.fTm1, cFeat.fTm2, cFeat.fTm3, cFeat.fTm4, cFeat.fTmD,
                         cFeat.nGCcnt1, cFeat.nGCcnt2, cFeat.nGCcnt3, cFeat.fGCcont1, cFeat.fGCcont2, cFeat.fGCcont3,
                         cFeat.fMFE1, cFeat.fMFE2, cFeat.fMFE3, cFeat.fMFE4, cFeat.fMFE5]

        sOutput      = ','.join([str(sOut) for sOut in list_sOut])
        OutFile.write('%s\n' % sOutput)

    #loop END: sKey
    OutFile.close()
#def END: make_output


if __name__ == '__main__':
    if len(sys.argv) == 1:
        main()
    else:
        function_name = sys.argv[1]
        function_parameters = sys.argv[2:]
        if function_name in locals().keys():
            locals()[function_name](*function_parameters)
        else:
            sys.exit('ERROR: function_name=%s, parameters=%s' % (function_name, function_parameters))
    # if END: len(sys.argv)
# if END: __name__    sInFile        = '%s/input/matplot_data.txt' % sDATA_DIR



