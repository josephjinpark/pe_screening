#!/home/hkim/anaconda3/bin/python

import os, sys, pickle, time, subprocess, json, re, regex, random
import numpy as np
import multiprocessing as mp

from Bio.SeqUtils import MeltingTemp as mt
from Bio.SeqUtils import GC as gc
from Bio.Seq import Seq

sTIME_STAMP = '%s'          % (time.ctime().replace(' ', '-').replace(':', '_'))
sVIENNA_BIN = '/data/scripts/bin/ViennaRNA-2.4.14'

class cFeatures:
    def __init__(self):

        cFeatures.sGuideKey  = ''
        cFeatures.sChrID       = ''
        cFeatures.sStrand      = ''
        cFeatures.nGenomicPos  = 0
        cFeatures.nEditIndex   = 0
        cFeatures.nPBSLen      = 0
        cFeatures.nRTTLen      = 0
        cFeatures.sPBSSeq      = ''
        cFeatures.sRTSeq       = ''
        cFeatures.sPegRNASeq   = ''
        cFeatures.sWTSeq       = ''
        cFeatures.sEditedSeq   = ''
        cFeatures.list_sSeqs   = []
        cFeatures.fTm1         = 0.0
        cFeatures.fTm2         = 0.0
        cFeatures.fTm2new      = 0.0
        cFeatures.fTm3         = 0.0
        cFeatures.fTm4         = 0.0
        cFeatures.fTmD         = 0.0
        cFeatures.fTm6         = 0.0
        cFeatures.fMFE1        = 0.0
        cFeatures.fMFE2        = 0.0
        cFeatures.fMFE3        = 0.0
        cFeatures.fMFE4        = 0.0
        cFeatures.fMFE5        = 0.0
        cFeatures.nGCcnt1      = 0
        cFeatures.nGCcnt2      = 0
        cFeatures.nGCcnt3      = 0
        cFeatures.fGCcont1     = 0.0
        cFeatures.fGCcont2     = 0.0
        cFeatures.fGCcont3     = 0.0
    #def END: __init__

def main():

    ## Constants ##
    dict_sSubType  = {'A':'t', 'T':'a', 'C':'g', 'G':'c'}
    nNickIndex     = 21
    nAltIndex      = 60 # 60nts --- Alt --- 60nts *0-based

    ## Input ##
    sANALYSISTAG   = '20210423-TEST'
    sINPUT_DIR     = '/extdata1/Jinman/pe_screening/input/2nd_structure'
    sInputFile     = '%s/DeepPE2_Feature_Extraction_%s.csv' % (sINPUT_DIR, sANALYSISTAG)
    list_cFeats    = load_input_data (sInputFile)
    print('list_cFeats', len(list_cFeats))
    ############

    ## Output ##
    sOUTPUT_DIR     = '/extdata1/Jinman/pe_screening/output/2nd_structure/%s' % sANALYSISTAG
    sTEMP_DIR       = '%s/temp' % sOUTPUT_DIR
    os.makedirs(sOUTPUT_DIR, exist_ok=True)
    os.makedirs(sTEMP_DIR, exist_ok=True)
    ############

    feature_extraction_mp([list_cFeats, nNickIndex, nAltIndex, ''])
    sys.exit()

    nFileCnt         = len(list_cFeats)
    nNoSplits        = 18
    list_nBins       = [[int(nFileCnt * (i + 0) / nNoSplits), int(nFileCnt * (i + 1) / nNoSplits)] for i in range(nNoSplits)]
    list_sParameters = []
    for nStart, nEnd in list_nBins:
        list_sSubSplits = list_cFeats[nStart:nEnd]
        sTempFile       = '%s/temp.%s.%s.output.txt' % (sTEMP_DIR, nStart, nEnd)

        print(list_sSubSplits, nNickIndex-2, sTempFile)

        list_sParameters.append([list_sSubSplits, nNickIndex-2, sTempFile])
    #loop END: nStart, nEnd

    p = mp.Pool(nNoSplits)
    p.map_async(feature_extraction_mp, list_sParameters).get()
#def END: main


def reverse_complement(sSeq):
    dict_sBases = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'N': 'N', 'U':'U', 'n':'',
                   '.': '.', '*': '*', 'a': 't', 'c': 'g', 'g': 'c', 't': 'a'}
    list_sSeq   = list(sSeq)  # Turns the sequence in to a gigantic list
    list_sSeq   = [dict_sBases[sBase] for sBase in list_sSeq]
    return ''.join(list_sSeq)[::-1]
#def END: reverse_complement


def load_input_data (sInFile):
    InFile       = open(sInFile, 'r')
    list_sOutput = []
    for i, sReadLine in enumerate(InFile):
        ## Format ##
        if i == 0: continue #SKIP HEADER

        list_sColumn          = sReadLine.strip('\n').split(',')

        cFeat                 = cFeatures()
        cFeat.sWTSeq          = list_sColumn[0]
        cFeat.sEditedSeq      = list_sColumn[1]
        cFeat.sAltKey         = list_sColumn[2]
        cFeat.sAltType        = cFeat.sAltKey[:-1].lower()
        cFeat.nAltLen         = int(cFeat.sAltKey[-1])

        list_sOutput.append(cFeat)
    #loop END: sReadLine
    InFile.close()
    return list_sOutput
#def END: load_input_data

def feature_extraction_mp (sParameters):
    list_cFeats = sParameters[0]
    nNickIndex  = sParameters[1]
    nAltIndex   = sParameters[2]
    sOutFile    = sParameters[3]

    print('Processing Output %s' % sOutFile)

    print(len(list_cFeats))
    for cFeat in list_cFeats:
        print(cFeat.sWTSeq, cFeat.sEditedSeq, cFeat.sAltKey, cFeat.nAltLen)
        ## Get all possible RT and PBS sequences ##
        get_all_RT_PBS(cFeat, nAltIndex)


        determine_seqs (cFeat, nNickIndex,  nAltIndex)
        #determine_secondary_structure(cFeat)
    #loop END: cFeat

    #make_output(list_cFeats, sOutFile)
#def END: feature_extraction_mp


def determine_seqs (cFeat, nNickIndex, nAltIndex):

    ## for Tm1
    cFeat.sForTm1 = reverse_complement(cFeat.sPBSSeq.replace('A','U'))
    #print(cFeat.sForTm1)

    ## for Tm2
    cFeat.sForTm2 = cFeat.sWTSeq[nNickIndex:nNickIndex+cFeat.nRTLen]
    #print(cFeat.sForTm2)

    ## for Tm2new
    cFeat.sForTm2new = cFeat.sWTSeq[nNickIndex:-2]
    #print(cFeat.sForTm2new)

    ## for Tm3
    list_sForTm3Seqs = []
    if cFeat.sRunType.startswith('SUB'):
        list_sSeqPairs = get_WT_Alt_seq_pairs(cFeat.sWTSeq[nNickIndex:-2], cFeat.sEditedSeq[nNickIndex:-2])
        for sEditedSeq, sWTSeq in list_sSeqPairs:

            list_sSeq1 = []
            list_sSeq2 = []
            for sNT1, sNT2 in zip(sEditedSeq, sWTSeq):
                if sNT1 != sNT2:
                    list_sSeq1.append(sNT1.lower())
                    list_sSeq2.append(sNT2.lower())
                else:
                    list_sSeq1.append(sNT1)
                    list_sSeq2.append(sNT2)
            #loop END: sNT1, sNT2
            sSeq1 = ''.join(list_sSeq1)
            sSeq2 = reverse_complement(''.join(list_sSeq2)[::-1])

            list_sForTm3Seqs.append([sSeq1, sSeq2])
        #loop END: sWTSeq, sEditedSeq
    else:

        nMinSize   = min([len(cFeat.sEditedSeq), len(cFeat.sWTSeq)])
        sTargetSeq = cFeat.sEditedSeq if len(cFeat.sEditedSeq) == nMinSize else cFeat.sWTSeq
        sAltNuc    = sTargetSeq[nNickIndex + cFeat.nEditIndex - 1]
        sAltSeq    = sTargetSeq[:nNickIndex + cFeat.nEditIndex -1] + reverse_complement(sAltNuc) + sTargetSeq[nNickIndex + cFeat.nEditIndex:]
        sCompSeq   = reverse_complement(sAltSeq[::-1])
        list_sForTm3Seqs.append([sTargetSeq[nNickIndex:], sCompSeq[nNickIndex:]])
    #if END:
    cFeat.sForTm3 = list_sForTm3Seqs

    ## for Tm4
    cFeat.sForTm4 = [reverse_complement(cFeat.sEditedSeq[nNickIndex:-2].replace('A','U')), cFeat.sEditedSeq[nNickIndex:-2]]
    #print(cFeat.sForTm4)
#def END: determine_seqs


def get_all_RT_PBS (cFeat, nAltIndex):

    nMinPBS         = 0
    nMaxPBS         = 17
    nMaxRT          = 40
    nMaxEditPosWin  = nMaxRT + 3 # Distance between PAM and mutation

    dict_sWinSize   = {'sub':{1:[nMaxRT - 1 - 3, 6], 2:[nMaxRT - 2 - 3, 6], 3:[nMaxRT - 3 - 3, 6]},
                       'ins':{1:[nMaxRT - 2 - 3, 6], 2:[nMaxRT - 3 - 3, 6], 3:[nMaxRT - 4 - 3, 6]},
                       'del':{1:[nMaxRT - 1 - 3, 6], 2:[nMaxRT - 1 - 3, 6], 3:[nMaxRT - 1 - 3, 6]}}


    dict_sRE        = {'+': '[ACGT]GG', '-': 'CC[ACGT]'}
    sTargetSeq      = cFeat.sEditedSeq

    for sStrand in ['+', '-']:

        sRE = dict_sRE[sStrand]

        for sReIndex in regex.finditer(sRE, sTargetSeq, overlapped=True):
            nIndexStart     = sReIndex.start()
            nIndexEnd       = sReIndex.end()
            sPAMSeq         = sTargetSeq[nIndexStart:nIndexEnd]
            nAltPosWin      = set_alt_position_window (sStrand, cFeat.sAltKey, nAltIndex, nIndexStart, nIndexEnd, cFeat.nAltLen)

            ## AltPosWin Filter ##
            if nAltPosWin <= 0:             continue
            if nAltPosWin > nMaxEditPosWin: continue

            nPAM_Nick = set_PAM_nicking_pos (sStrand, cFeat.sAltType, cFeat.nAltLen, nAltIndex, nIndexStart, nIndexEnd)

            print('nMaxEditPosWin', nMaxEditPosWin, nPAM_Nick)
            print(nIndexStart, nIndexEnd)
            print(cFeat.sWTSeq)
            print(sTargetSeq)
            print('%s%s' % (' '*nAltIndex, '*'*cFeat.nAltLen))

            print('%s%s\t%s' % ('-'*nIndexStart, sPAMSeq, nAltPosWin))

            '''
            ## Substitution Filter ##
            if sAltKey.startswith('sub'):
                if nAltPosWin < -3: continue
            #########################

            ### Indel Filter ###
            if sAltKey.startswith('del') or sAltKey.startswith('ins'):
                if sStrand == '+':
                    if nAltPosWin < -4: continue
                else:
                    if nAltPosWin < (-4 + cVCF.nAltLen): continue
            #####################
            #################


            if not check_PAM_window(dict_sWinSize, sStrand, nIndexStart, nIndexEnd, cFeat.sAltType, cFeat.nAltLen, nAltIndex): continue


            if sStrand == '+':
                sGuideSeq       = sForGuideSeq[nIndexStart - nGuideUp:nIndexEnd + nGuideDown]
                nGenomicS_Guide = cVCF.nStartPos + nIndexStart - nGuideUp
                nGenomicE_Guide = cVCF.nStartPos + nIndexEnd + nGuideDown - 1

            else:
                sGuideSeq       = reverse_complement(sForGuideSeq[nIndexStart - nGuideDown:nIndexEnd + nGuideUp])
                nGenomicS_Guide = cVCF.nStartPos + nIndexStart - nGuideDown
                nGenomicE_Guide = cVCF.nStartPos + nIndexEnd + nGuideUp - 1
            #if END: sStrand

            #nFrameNo       = determine_PAM_ORF (cVCF, sStrand, dict_sORFData, sPAMSeq, nGenomicS_PAM, nGenomicE_PAM)
            #if nFrameNo is None: continue

            sPAM_check     = sGuideSeq[nGuideUp:nGuideUp+len(sPAMSeq)]

            PAM_pos_in_guide_check(cVCF, sStrand, sGuideSeq, sPAM_check, sPAMSeq, sForGuideSeq, sForTempSeq)

            sPAMKey        = '%s,%s,%s,%s,%s,%s,%s' \
                             % (sStrand, sGuideSeq, nAltPosWin, cVCF.sAltNuc,
                                sPAMSeq, nFrameNo, cVCF.sGeneSym)

            #if bTestRun: print(sGuideSeq, sPAMSeq, sStrand, nIndexStart, nIndexEnd, nPAM_Nick, nAltPosWin, cVCF.sRefNuc, cVCF.sAltNuc)

            if sPAMKey not in cVCF.dict_PAM:
                cVCF.dict_PAM[sPAMKey] = ''

            cVCF.dict_PAM[sPAMKey] = determine_PBS_RT_seq(sPool, sAltKey, sStrand, nMinPBS, nMaxPBS, nMaxRT, nSetPBSLen, nSetRTLen, cVCF.nAltLen, nAltIndex, nPAM_Nick, nAltPosWin, sForTempSeq, bTestRun)
            nCnt1, nCnt2           = len(cVCF.dict_PAM[sPAMKey][0]), len(cVCF.dict_PAM[sPAMKey][1])

            if nAltPosWin not in dict_sStats:
                dict_sStats[nAltPosWin] = 0
            dict_sStats[nAltPosWin] += 1

            sCheckKey = '%s,%s' % (nCnt1, nCnt2)
            if sCheckKey not in dict_sStats2:
                dict_sStats2[sCheckKey] = 0
            dict_sStats2[sCheckKey] += 1
            '''

        #loop END: sReIndex
    #loop END: sStrand

#def END: get_all_RT_PBS


def set_alt_position_window (sStrand, sAltKey, nAltIndex, nIndexStart, nIndexEnd, nAltLen):

    if sStrand == '+':

        if sAltKey.startswith('sub'):   return (nAltIndex + 1) - (nIndexStart - 3)
        else:                           return (nAltIndex + 1) - (nIndexStart - 3) + 1

    else:
        if sAltKey.startswith('sub'):
            return nIndexEnd - nAltIndex + 3 - (nAltLen - 1)

        elif sAltKey.startswith('del'):
            return nIndexEnd - nAltIndex + 3 - nAltLen

        else:
            return nIndexEnd - nAltIndex + 3
        #if END:
    #if END:
#def END: set_alt_position_window_v2


def set_PAM_nicking_pos(sStrand, sAltType, nAltLen, nAltIndex, nIndexStart, nIndexEnd):

    if sStrand == '-':
        #if nIndexEnd <= nAltIndex:
        if sAltType == 'del':
                nPAM_Nick = nIndexEnd + 3 - nAltLen

        elif sAltType == 'ins':
                nPAM_Nick = nIndexEnd + 3 + nAltLen

        elif sAltType == 'sub':
            nPAM_Nick = nIndexEnd + 3
    else:
        if nIndexStart >= nAltIndex:
            if sAltType == 'del':
                    nPAM_Nick = nIndexStart - 3

            elif sAltType == 'ins':
                    nPAM_Nick = nIndexStart - 3

            elif sAltType == 'sub':
                nPAM_Nick = nIndexStart - 3

        else:
            nPAM_Nick = nIndexStart - 3
        #if END: nIndexStart
    #if END: sStrand
    return nPAM_Nick
#def END: set_PAM_Nicking_Pos

def check_PAM_window (dict_sWinSize, sStrand, nIndexStart, nIndexEnd, sAltType, nAltLen, nAltIndex):

    nUp, nDown = dict_sWinSize[sAltType][nAltLen]

    if sStrand == '+':
        nPAMCheck_min = nAltIndex - nUp + 1
        nPAMCheck_max = nAltIndex + nDown + 1
    else:
        nPAMCheck_min = nAltIndex - nDown + 1
        nPAMCheck_max = nAltIndex + nUp + 1
    #if END:

    if nIndexStart < nPAMCheck_min or nIndexEnd > nPAMCheck_max:
        return 0
    else:
        return 1
#def END: check_PAM_window


def determine_PBS_RT_seq (sPool, sAltKey, sStrand, nMinPBS, nMaxPBS, nMaxRT, nSetPBSLen, nSetRTLen, nAltLen, nAltIndex, nPAM_Nick, nAltPosWin, sForTempSeq, bTestRun):

    dict_sPBS = {}
    dict_sRT  = {}

    list_nPBSLen = [nNo + 1 for nNo in range(nMinPBS, nMaxPBS)]
    for nPBSLen in list_nPBSLen:

        ## Set PBS Length ##
        if nSetPBSLen:
            if nPBSLen != nSetPBSLen: continue

        if sStrand == '+':
            nPBSStart  = nPAM_Nick - nPBSLen  # 5' -> PamNick
            nPBSEnd    = nPAM_Nick
            sPBSSeq    = sForTempSeq[nPBSStart:nPBSEnd]

            #if bTestRun: print('>' * nPBSStart + sPBSSeq, nPBSStart, nPBSEnd, len(sPBSSeq))
        else:
            nPBSStart  = nPAM_Nick
            nPBSEnd    = nPAM_Nick + nPBSLen  # 5' -> PamNick
            sPBSSeq    = reverse_complement(sForTempSeq[nPBSStart:nPBSEnd])

            #if bTestRun: print('>' * nPBSStart  + sPBSSeq, nPBSStart, nPBSEnd, len(sPBSSeq))
        #if END: sStrand

        sKey       = len(sPBSSeq)
        if sKey not in dict_sPBS:
            dict_sPBS[sKey] = ''
        dict_sPBS[sKey] = sPBSSeq
    #loop END: nPBSLen

    if sStrand == '+':
        list_nRTPos  = [nNo + 1 for nNo in range(nAltIndex + nAltLen, (nPAM_Nick + nMaxRT))]
    else:
        list_nRTPos  = [nNo for nNo in range(nPAM_Nick - nMaxRT, nAltIndex + nAltLen)]

    for nRTPos in list_nRTPos:

        if sStrand == '+':
            nRTStart   = nPAM_Nick   # PamNick -> 3'
            nRTEnd     = nRTPos
            sRTSeq     = sForTempSeq[nRTStart:nRTEnd]

            #if bTestRun: print('>' * nPAM_Nick + sRTSeq, nRTStart, nRTEnd, len(sRTSeq))
        else:
            nRTStart   = nRTPos
            nRTEnd     = nPAM_Nick   # PamNick -> 3'
            sRTSeq     = reverse_complement(sForTempSeq[nRTStart:nRTEnd])

            if not sRTSeq: continue

            #if bTestRun: print('>' * nRTStart + sRTSeq, nRTStart, nRTEnd, len(sRTSeq))
        #if END: sStrand

        sKey = len(sRTSeq)

        ## Set RT Length ##
        if nSetRTLen:
            if sKey != nSetRTLen: continue

        if sKey > nMaxRT: continue
        if sKey < abs(nAltIndex - nPAM_Nick): continue

        if sPool == 'Therapy':
            if sAltKey.startswith('del'):
                if sKey < nAltPosWin + nAltLen: continue
        else:
            if sAltKey.startswith('ins'):
                if sKey < nAltPosWin + nAltLen: continue
        if sKey not in dict_sRT:
            dict_sRT[sKey] = ''
        dict_sRT[sKey] = sRTSeq
    #loop END: nRTPos

    return [dict_sPBS, dict_sRT]
#def END: determine_PBS_RT_seq


def determine_secondary_structure (cFeat):

    determine_Tm(cFeat)
    #determine_GC(cFeat)
    determine_MFE(cFeat)
#def END: determine_secondary_structure


def determine_Tm (cFeat):
    ## Tm1 DNA/RNA mm1 ##
    #print('Tm1', cFeat.sForTm1)
    cFeat.fTm1 = mt.Tm_NN(seq=Seq(cFeat.sForTm1), nn_table=mt.R_DNA_NN1)
    #print('Tm1', cFeat.fTm1)

    ## Tm2 DNA/DNA mm0 ##
    #print('Tm2', cFeat.sForTm2)
    cFeat.fTm2 = mt.Tm_NN(seq=Seq(cFeat.sForTm2), nn_table=mt.DNA_NN3)
    #print('Tm2', cFeat.fTm2)

    ## Tm2new DNA/DNA mm0 ##
    #print('Tm2new', cFeat.sForTm2new)
    cFeat.fTm2new = mt.Tm_NN(seq=Seq(cFeat.sForTm2new), nn_table=mt.DNA_NN3)
    #print('Tm2new', cFeat.fTm2new)

    ## Tm3 DNA/DNA mm1 ##
    if not cFeat.sForTm3:
        cFeat.fTm3    = 0
        cFeat.fTm3min = 0
        cFeat.fTm3max = 0
        cFeat.fTm3avg = 0
        cFeat.fTmD    = 0

    else:
        list_fTm3 = []
        for sSeq1, sSeq2 in cFeat.sForTm3:
            try:
                fTm3 = mt.Tm_NN(seq=sSeq1, c_seq=sSeq2, nn_table=mt.DNA_NN3)
            except ValueError: continue

            list_fTm3.append(fTm3)
        #loop END: sSeq1, sSeq2
        if list_fTm3:
            cFeat.fTm3min = min(list_fTm3)
            cFeat.fTm3max = max(list_fTm3)
            cFeat.fTm3avg = np.mean(list_fTm3)
        else:
            cFeat.fTm3min = 0
            cFeat.fTm3max = 0
            cFeat.fTm3avg = 0
            cFeat.fTmD    = 0
    #if END:
    #print('Tm3', cFeat.sForTm3)
    #print(cFeat.fTm3min, cFeat.fTm3max, cFeat.fTm3avg)

    #Tm4 - revcom(AAGTcGATCC(RNA version)) + AAGTcGATCC
    #print('Tm4', cFeat.sForTm4)
    cFeat.fTm4 = mt.Tm_NN(seq=Seq(cFeat.sForTm4[0]),  nn_table=mt.R_DNA_NN1)
    #print('Tm4', cFeat.fTm4)

    #Tm5 - Tm3 - Tm2
    cFeat.fTmD = cFeat.fTm3 - cFeat.fTm2

    #Tm6 -
    #print('Tm6', cFeat.sForTm4[1], reverse_complement(cFeat.sForTm4[1])[::-1])
    cFeat.fTm6 = mt.Tm_NN(seq=Seq(cFeat.sForTm4[1]), c_seq=Seq(reverse_complement(cFeat.sForTm4[1])[::-1]), nn_table=mt.DNA_NN3)
    #print('Tm6', cFeat.fTm6)
#def END: determine_Tm


def get_WT_Alt_seq_pairs (sRTseqWT, sRTseqAlt):
    nMMCnt       = len(['%s>%s' % (sWT, sAlt)for sWT, sAlt in zip(sRTseqWT, sRTseqAlt) if sWT != sAlt])
    list_sOutput = []

    if nMMCnt > 1:

        list_nMMIndex = [i for i, (sWT, sAlt) in enumerate(zip(sRTseqWT, sRTseqAlt)) if sWT != sAlt]
        for i in list_nMMIndex:
            sTempSeq = ''.join([sRTseqWT[n] if n != i else sRTseqAlt[i] for n in range(len(sRTseqWT))])
            list_sOutput.append([sTempSeq, sRTseqWT])
        #loop END: i

    else:
        list_sOutput.append([sRTseqAlt, sRTseqWT])
    #if END:

    return list_sOutput

#def END: get_WT_Alt_seq_pairs


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


def determine_MFE (cFeat):

    sScaffoldSeq    = 'GTTTTAGAGCTAGAAATAGCAAGTTAAAATAAGGCTAGTCCGTTATCAACTTGAAAAAGTGGCACCGAGTCGGTGC'
    sPolyTSeq       = 'TTTTTT'

    ## Set GuideRNA seq ##
    sGuideSeq       = 'G' + cFeat.sGuideSeq[6:24]

    #MFE_1 - pegRNA:
    sInputSeq             = sGuideSeq + sScaffoldSeq + reverse_complement(cFeat.sPBSSeq + cFeat.sRTSeq) + sPolyTSeq
    sDBSeq, cFeat.fMFE1   = get_dotbracket_and_MFE(sInputSeq, 0.0)
    #print(sDBSeq, cFeat.fMFE1)

    #MFE_2 - pegRNA without spacer
    sInputSeq             = sScaffoldSeq + reverse_complement(cFeat.sPBSSeq + cFeat.sRTSeq) + sPolyTSeq
    sDBSeq, cFeat.fMFE2   = get_dotbracket_and_MFE(sInputSeq, 0.0)
    #print(sDBSeq, cFeat.fMFE2)

    #MFE_3 - RT + PBS + PolyT
    sInputSeq             = reverse_complement(cFeat.sPBSSeq + cFeat.sRTSeq) + sPolyTSeq
    sDBSeq, cFeat.fMFE3   = get_dotbracket_and_MFE(sInputSeq, 0.0)
    #print(sDBSeq, cFeat.fMFE3)

    #MFE_4 - spacer only
    sInputSeq             = sGuideSeq
    sDBSeq, cFeat.fMFE4   = get_dotbracket_and_MFE(sInputSeq, 0.0)
    #print(sDBSeq, cFeat.fMFE4)

    #MFE_5 - Spacer + Scaffold
    sInputSeq             = sGuideSeq + sScaffoldSeq
    sDBSeq, cFeat.fMFE5   = get_dotbracket_and_MFE(sInputSeq, 0.0)
    #print(sDBSeq, cFeat.fMFE5)

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


def make_output (list_cFeats, sOutFile):

    OutFile = open(sOutFile, 'w')
    sHeader = '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' % ('GuideKey', 'Tm1', 'Tm2', 'Tm2new', 'Tm3min',
                                                                  'Tm3max', 'Tm3avg', 'Tm4', 'TmD', 'Tm6', 'MFE1', 'MFE2',
                                                                  'MFE3', 'MFE4', 'MFE5')
    OutFile.write(sHeader)
    for cFeat in list_cFeats:
        list_sOut = [cFeat.sGuideKey, cFeat.fTm1, cFeat.fTm2, cFeat.fTm2new, cFeat.fTm3min, cFeat.fTm3max, cFeat.fTm3avg,
                     cFeat.fTm4, cFeat.fTmD, cFeat.fTm6,
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
# if END: __name__



