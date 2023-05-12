#!/home/jinmanlab/bin/python3

import os, sys, pickle, time, subprocess, json, re, regex, random
import numpy as np
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
        #INDEX,Chr,Strand,Genomic Position,Type,PE_edit_type,Mutation length,Edit Position,PBS length,RTT length,PBS,RT,pegRNA,WTseq,EditedSeq
        if i == 0: continue #SKIP HEADER

        list_sColumn          = sReadLine.strip('\n').split(',')

        cFeat                 = cFeatures()
        cFeat.sGuideKey       = list_sColumn[0].upper()
        cFeat.sLibrary        = cFeat.sGuideKey.split('_')[0]
        cFeat.sRunType        = cFeat.sGuideKey.split('_')[1].replace('PEGRNA', '')
        cFeat.nMutLen         = int(cFeat.sGuideKey.split('_')[1].replace('PEGRNA', '')[-1])
        cFeat.sChrID          = list_sColumn[1]
        cFeat.sStrand         = list_sColumn[2]
        cFeat.nGenomicPos     = list_sColumn[3]
        cFeat.nEditIndex      = int(list_sColumn[4])
        cFeat.nPBSLen         = int(list_sColumn[5])
        cFeat.nRTLen          = int(list_sColumn[6])
        cFeat.sPBSSeq         = list_sColumn[7]
        cFeat.sRTSeq          = list_sColumn[8]
        cFeat.sGuideSeq       = list_sColumn[9]
        cFeat.sWTSeq          = list_sColumn[10]
        cFeat.sEditedSeq      = list_sColumn[11]

        list_sOutput.append(cFeat)
    #loop END: sReadLine
    InFile.close()
    return list_sOutput
#def END: load_input_data


def determine_seqs (cFeat, nNickIndex):

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
        print(sTargetSeq[:nNickIndex + cFeat.nEditIndex-1], sAltNuc, sTargetSeq[nNickIndex + cFeat.nEditIndex:])
        print(sAltSeq[:nNickIndex + cFeat.nEditIndex-1], sAltNuc, sAltSeq[nNickIndex + cFeat.nEditIndex:])
        print(sTargetSeq[nNickIndex:])
        print(sCompSeq[nNickIndex:])

        sys.exit()

        '''
        if cFeat.sLibrary == 'MODELING':
            
            
            
            sAltSeq   = cFeat.sEditedSeq if cFeat.sRunType.startswith('INS') else cFeat.sWTSeq
            sAltNuc   = sAltSeq[nNickIndex+cFeat.nEditIndex:nNickIndex+cFeat.nEditIndex+cFeat.nMutLen]

            sEditSeq  = cFeat.sWTSeq if cFeat.sRunType.startswith('INS') else cFeat.sEditedSeq
            sFinalSeq = reverse_complement(sEditSeq[:nNickIndex+cFeat.nEditIndex])[::-1] + sAltNuc + reverse_complement(sEditSeq[nNickIndex+cFeat.nEditIndex:])[::-1]

            #print('sEditSeq', sEditSeq)
            #print('PT1', sEditSeq[nNickIndex+cFeat.nEditIndex:])
            #print('PT2', sAltNuc)
            #print('PT3', sEditSeq[:nNickIndex+cFeat.nEditIndex])
            #print('sFinalSeq', sFinalSeq)

            if cFeat.sRunType.startswith('INS'):
                list_sForTm3Seqs.append([cFeat.sEditedSeq, sFinalSeq])
            else:
                list_sForTm3Seqs.append([sFinalSeq, cFeat.sWTSeq])


        else:
            sAltSeq    = cFeat.sWTSeq if cFeat.sRunType.startswith('INS') else cFeat.sEditedSeq
            sAltNuc    = sAltSeq[nNickIndex+cFeat.nEditIndex:nNickIndex+cFeat.nEditIndex+cFeat.nMutLen]

            sEditSeq  = cFeat.sEditedSeq if cFeat.sRunType.startswith('INS') else cFeat.sWTSeq
            sFinalSeq = sEditSeq[:nNickIndex+cFeat.nEditIndex] + sAltNuc + sEditSeq[nNickIndex+cFeat.nEditIndex:]
            #print(sEditSeq[:nNickIndex+cFeat.nEditIndex])
            #print(sAltNuc)
            #print(nNickIndex+cFeat.nEditIndex+cFeat.nMutLen)
            #print(sEditSeq[nNickIndex+cFeat.nEditIndex:])

            if cFeat.sRunType.startswith('INS'):
                list_sForTm3Seqs.append([sFinalSeq, cFeat.sWTSeq])
            else:
                list_sForTm3Seqs.append([cFeat.sEditedSeq, sFinalSeq])

        #print(cFeat.sGuideKey)
        #print(cFeat.sLibrary)
        #print(cFeat.sEditedSeq, cFeat.sWTSeq)
        #print(list_sForTm3Seqs)'''


    cFeat.sForTm3 = list_sForTm3Seqs

    ## for Tm4
    cFeat.sForTm4 = [reverse_complement(cFeat.sEditedSeq[nNickIndex:-2].replace('A','U')), cFeat.sEditedSeq[nNickIndex:-2]]
    #print(cFeat.sForTm4)
#def END: determine_seqs_pt1




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
    sHeader = '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' % ('GuideKey', 'Tm1', 'Tm2', 'Tm2new', 'Tm3min',
                                                            'Tm3max', 'Tm3avg', 'Tm4', 'TmD', 'MFE1', 'MFE2',
                                                            'MFE3', 'MFE4', 'MFE5')
    OutFile.write(sHeader)
    for cFeat in list_cFeats:
        list_sOut = [cFeat.sGuideKey, cFeat.fTm1, cFeat.fTm2, cFeat.fTm2new, cFeat.fTm3min, cFeat.fTm3max,cFeat.fTm3avg,
                     cFeat.fTm4, cFeat.fTmD,
                     cFeat.fMFE1, cFeat.fMFE2, cFeat.fMFE3, cFeat.fMFE4, cFeat.fMFE5]

        sOutput      = ','.join([str(sOut) for sOut in list_sOut])
        OutFile.write('%s\n' % sOutput)

    #loop END: sKey
    OutFile.close()
#def END: make_output

def main():

    ## Constants ##
    dict_sSubType  = {'A':'t', 'T':'a', 'C':'g', 'G':'c'}
    nNickIndex     = 21

    ## Input ##
    sINPUT_DIR     = '/data/scripts/pe_screening/input/secondary_structure'
    sInputFile     = '%s/DeepPE2_Feature_Extraction_201222.csv' % sINPUT_DIR
    list_cFeats    = load_input_data (sInputFile)
    ############

    ## Output ##
    sOUTPUT_DIR    = '/data/scripts/pe_screening/output/secondary_structure'
    os.makedirs(sOUTPUT_DIR, exist_ok=True)
    sOutFile       = '%s/20201223_formodeling.output.pt1.csv' % sOUTPUT_DIR
    ############


    dict_sOutput = {}
    for cFeat in list_cFeats[:1]:
        #if cFeat.sGuideKey != 'MODELING_INS3PEGRNA_76001': continue
        #if cFeat.sGuideKey != 'MODELING_DEL3PEGRNA_44809': continue
        #if cFeat.sGuideKey != 'THERAPY_INS3PEGRNA_451225': continue
        #if cFeat.sGuideKey != 'THERAPY_DEL3PEGRNA_339217': continue

        print(cFeat.sGuideKey, cFeat.nEditIndex, cFeat.sPBSSeq, cFeat.sRTSeq, cFeat.sGuideSeq)
        determine_seqs (cFeat, nNickIndex-2)
        determine_secondary_structure(cFeat)
    #loop END: sTarSeq

    make_output(list_cFeats, sOutFile)
#def END: main


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



