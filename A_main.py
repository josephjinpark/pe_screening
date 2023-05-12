#!/home/jinmanlab/bin/python3

import os, sys, pickle, time, subprocess, json, re, regex, random
import numpy as np
import matplotlib as mpl
import multiprocessing as mp

mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import Locator

from Bio import SeqIO
from Bio import pairwise2
from Bio.pairwise2 import format_alignment

sGENOME     = 'hg38'
sSRC_DIR    = '/scripts/HKLab_Scripts'
sDATA_DIR   = '/data/scripts'
sREF_DIR    = '/data/reference_genome'
sGENOME_DIR = '%s/%s'       % (sREF_DIR, sGENOME)
sCHRSEQ_DIR = '%s/Splited'  % sGENOME_DIR

sTIME_STAMP = '%s'          % (time.ctime().replace(' ', '-').replace(':', '_'))
sLIFTOVER   = '%s/liftOver' % sREF_DIR
sFLASH      = '%s/bin/FLASH-1.2.11-Linux-x86_64/flash' % sDATA_DIR
nLINE_CNT_LIMIT = 500000 ## for split files

list_sCHRIDs_MM = [str(x) for x in range(1, 20)]
list_sCHRIDs_HG = [str(x) for x in range(1, 23)] + ['X', 'Y', 'M']

dict_sAA_TABLE  = {'ATA': 'I', 'ATC': 'I', 'ATT': 'I', 'ATG': 'M',
                  'ACA': 'T', 'ACC': 'T', 'ACG': 'T', 'ACT': 'T',
                  'AAC': 'N', 'AAT': 'N', 'AAA': 'K', 'AAG': 'K',
                  'AGC': 'S', 'AGT': 'S', 'AGA': 'R', 'AGG': 'R',
                  'CTA': 'L', 'CTC': 'L', 'CTG': 'L', 'CTT': 'L',
                  'CCA': 'P', 'CCC': 'P', 'CCG': 'P', 'CCT': 'P',
                  'CAC': 'H', 'CAT': 'H', 'CAA': 'Q', 'CAG': 'Q',
                  'CGA': 'R', 'CGC': 'R', 'CGG': 'R', 'CGT': 'R',
                  'GTA': 'V', 'GTC': 'V', 'GTG': 'V', 'GTT': 'V',
                  'GCA': 'A', 'GCC': 'A', 'GCG': 'A', 'GCT': 'A',
                  'GAC': 'D', 'GAT': 'D', 'GAA': 'E', 'GAG': 'E',
                  'GGA': 'G', 'GGC': 'G', 'GGG': 'G', 'GGT': 'G',
                  'TCA': 'S', 'TCC': 'S', 'TCG': 'S', 'TCT': 'S',
                  'TTC': 'F', 'TTT': 'F', 'TTA': 'L', 'TTG': 'L',
                  'TAC': 'Y', 'TAT': 'Y', 'TAA': '_', 'TAG': '_',
                  'TGC': 'C', 'TGT': 'C', 'TGA': '_', 'TGG': 'W', }


## region Classes

class cAbsoluteCNV: pass
class cGUIDESData: pass
class cPEData: pass
class cGuideKeyInfo: pass

## region class cVCFData

class MinorSymLogLocator(Locator):
    """
    Dynamically find minor tick positions based on the positions of
    major ticks for a symlog scaling.
    """

    def __init__(self, linthresh):
        """
        Ticks will be placed between the major ticks.
        The placement is linear for x between -linthresh and linthresh,
        otherwise its logarithmically
        """
        self.linthresh = linthresh

    def __call__(self):
        'Return the locations of the ticks'
        majorlocs = self.axis.get_majorticklocs()

        # iterate through minor locs
        minorlocs = []

        # handle the lowest part
        for i in range(1, len(majorlocs)):
            majorstep = majorlocs[i] - majorlocs[i - 1]
            if abs(majorlocs[i - 1] + majorstep / 2) < self.linthresh:
                ndivs = 10
            else:
                ndivs = 9
            minorstep = majorstep / ndivs
            locs = np.arange(majorlocs[i - 1], majorlocs[i], minorstep)[1:]
            minorlocs.extend(locs)

        return self.raise_if_exceeds(np.array(minorlocs))

    def tick_values(self, vmin, vmax):
        raise NotImplementedError('Cannot get tick locations for a '
                                  '%s type.' % type(self))


class cVCFData:
    def __init__(self):
        self.sPatID = ''
        self.sChrID = ''
        self.nPos = 0
        self.sDBSNP_ID = ''
        self.sRefNuc = ''
        self.sAltNuc = ''
        self.fQual = 0.0
        self.sFilter = ''
        self.sInfo = ''
        self.sFormat = ''
        self.list_sMisc = []

        # Extra
        self.nClusterID = 0

    #def END: __int__


def cVCF_parse_vcf_files(sVCFFile):
    if not os.path.isfile(sVCFFile):
        sys.exit('File Not Found %s' % sVCFFile)

    list_sOutput = []
    InFile = open(sVCFFile, 'r')

    for sReadLine in InFile:
        # File Format
        # Column Number:     | 0       | 1        | 2          | 3       | 4
        # Column Description:| sChrID  | nPos     | sDBSNP_ID  | sRefNuc | sAltNuc
        # Column Example:    | 1       | 32906558 | rs79483201 | T       | A
        # Column Number:     | 5       | 6        | 7          | 8              | 9./..
        # Column Description:| fQual   | sFilter  | sInfo      | sFormat        | sSampleIDs
        # Column Example:    | 5645.6  | PASS     | .          | GT:AD:DP:GQ:PL | Scores corresponding to sFormat

        if sReadLine.startswith('#'): continue  # SKIP Information Headers
        list_sColumn = sReadLine.strip('\n').split('\t')

        cVCF = cVCFData()
        cVCF.sChrID = 'chr%s' % list_sColumn[0]

        try:
            cVCF.nPos = int(list_sColumn[1])
        except ValueError:
            continue

        cVCF.sDBSNP_ID  = list_sColumn[2]
        cVCF.sRefNuc    = list_sColumn[3]
        cVCF.sAltNuc    = list_sColumn[4]
        cVCF.fQual      = float(list_sColumn[5]) if list_sColumn[5] != '.' else list_sColumn[5]
        cVCF.sFilter    = list_sColumn[6]
        cVCF.sInfo  = list_sColumn[7]

        dict_sInfo = dict([sInfo.split('=') for sInfo in cVCF.sInfo.split(';') if len(sInfo.split('=')) == 2])

        try:
            cVCF.sAlleleFreq = float(dict_sInfo['AF_raw'])
        except ValueError:
            cVCF.sAlleleFreq = np.mean([float(f) for f in dict_sInfo['AF_raw'].split(',')])

        list_sOutput.append(cVCF)
    #loop END: sReadLine
    InFile.close()

    return list_sOutput
#def END: cVCF_parse_vcf_files


def parse_vcf_stdout2(sStdOut):
    list_sOutput = []
    for sReadLine in sStdOut:
        # File Format
        # Column Number:     | 0       | 1        | 2          | 3       | 4
        # Column Description:| sChrID  | nPos     | sDBSNP_ID  | sRefNuc | sAltNuc
        # Column Example:    | chr13   | 32906558 | rs79483201 | T       | A
        # Column Number:     | 5       | 6        | 7          | 8              | 9./..
        # Column Description:| fQual   | sFilter  | sInfo      | sFormat        | sSampleIDs
        # Column Example:    | 5645.6  | PASS     | .          | GT:AD:DP:GQ:PL | Scores corresponding to sFormat
        ##FORMAT=<ID=AD,Number=.,Type=Integer,Description="Allelic depths for the ref and alt alleles in the order listed">
        ##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Approximate read depth (reads with MQ=255 or with bad mates are filtered)">
        ##FORMAT=<ID=GQ,Number=1,Type=Integer,Description="Genotype Quality">
        ##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
        ##FORMAT=<ID=PL,Number=G,Type=Integer,Description="Normalized, Ph
        sReadLine       = str(sReadLine, 'UTF-8')
        list_sColumn    = sReadLine.strip('\n').split('\t')
        cVCF            = cVCFData()

        if list_sColumn[0] == 'MT': continue
        if list_sColumn[0].startswith('<GL00'): continue

        dict_sChrKey = {'X': '23', 'Y': '24'}
        cVCF.sChrID = 'chr%s' % list_sColumn[0]

        if list_sColumn[0] in ['X', 'Y']:
            cVCF.nChrID = int(dict_sChrKey[list_sColumn[0]])
        else:
            cVCF.nChrID = int(list_sColumn[0])
        cVCF.nPos       = int(list_sColumn[1])
        cVCF.sDBSNP_ID  = list_sColumn[2]
        cVCF.sRefNuc    = list_sColumn[3]
        cVCF.sAltNuc    = list_sColumn[4]
        cVCF.fQual      = float(list_sColumn[5]) if list_sColumn[5] != '.' else list_sColumn[5]
        cVCF.sFilter    = list_sColumn[6]
        cVCF.sInfo      = list_sColumn[7]
        dict_sInfo      = dict([sInfo.split('=') for sInfo in cVCF.sInfo.split(';') if len(sInfo.split('=')) == 2])

        try:
            cVCF.fAlleleFreq = float(dict_sInfo['AF_raw'])
        except ValueError:
            cVCF.fAlleleFreq = np.mean([float(f) for f in dict_sInfo['AF_raw'].split(',')])

        list_sOutput.append(cVCF)
    #loop END: sReadLine

    return list_sOutput
#def END: parse_vcf_stdout2


## endregion

## region class cFasta
re_nonchr = re.compile('[^a-zA-Z]')


class cFasta:
    def __init__(self, sRefFile):

        # V-S Check
        if not os.path.isfile(sRefFile):
            sys.exit('(): File does not exist')

        self.InFile = open(sRefFile, 'r')
        self.sChrIDList = []
        self.nChromLen = []
        self.nSeekPos = []
        self.nLen1 = []
        self.nLen2 = []

        # V-S Check
        if not os.path.isfile('%s.fai' % sRefFile):
            sys.exit('.fai file does not exist')

        InFile = open('%s.fai' % sRefFile, 'r')
        for sLine in InFile:
            list_sColumn = sLine.strip('\n').split()  # Goes backwards, -1 skips the new line character

            self.sChrIDList.append(list_sColumn[0])
            self.nChromLen.append(int(list_sColumn[1]))
            self.nSeekPos.append(int(list_sColumn[2]))
            self.nLen1.append(int(list_sColumn[3]))
            self.nLen2.append(int(list_sColumn[4]))
        #loop END: sLINE
        InFile.close()
        self.sType = []

    #def END: __init_

    def fetch(self, sChrom, nFrom=None, nTo=None, sStrand='+'):
        assert sChrom in self.sChrIDList, sChrom
        nChrom = self.sChrIDList.index(sChrom)

        if nFrom == None: nFrom = 0
        if nTo == None: nTo = self.nChromLen[nChrom]
        # if nTo >= self.nChromLen[nChrom]: nTo = self.nChromLen[nChrom]-1

        assert (0 <= nFrom) and (nFrom < nTo) and (nTo <= self.nChromLen[nChrom])

        nBlank = self.nLen2[nChrom] - self.nLen1[nChrom]

        nFrom = int(nFrom + (nFrom / self.nLen1[nChrom]) * nBlank)  # Start Fetch Position

        nTo = int(nTo + (nTo / self.nLen1[nChrom]) * nBlank)  # End Fetch Position

        self.InFile.seek(self.nSeekPos[nChrom] + nFrom)  # Get Sequence

        sFetchedSeq = re.sub(re_nonchr, '', self.InFile.read(nTo - nFrom))

        if sStrand == '+': return sFetchedSeq
        elif sStrand == '-': return reverse_complement(sFetchedSeq)
        else: sys.exit('Invalid Strand')
    #def END: fetch


#class END: Fasta
## endregion

## region class cRefSeqData

class cRefSeqData:
    def __init__(self):
        # Initalization of instances variables
        self.sGeneSym       = ''
        self.sNMID          = ''
        self.sChrID         = ''
        self.nChrID         = ''
        self.sStrand        = ''
        self.nTxnStartPos   = 0
        self.nTxnEndPos     = 0
        self.nORFStartPos   = 0
        self.nORFEndPos     = 0
        self.nExonCount     = 0
        self.list_nExonS    = []
        self.list_nExonE    = []
        self.fFoldChange    = []
        self.bRemovalFlag   = 0
        self.s5UTRSeq       = ''
        self.sORFSeq        = ''
        self.s3UTRSeq       = ''
        self.n5UTRLen       = 0
        self.nORFLen        = 0
        self.n3UTRLen       = 0
        self.bDownReg       = 0
        self.n5UTRStartList = []
        self.n5UTREndList   = []  # For '+' Stand, coding start pos is the end
        self.nORFStartList  = []
        self.nORFEndList    = []
        self.n3UTRStartList = []  # For '-' Stand, coding end pos is the start
        self.n3UTREndList   = []

    #def END: __init__


def parse_refflat_line(sInFile, sOutput, sKeyType):
    list_sOutput = []
    InFile = open(sInFile, 'r')

    for sReadLine in InFile:

        # refFlat file format                                                            Txn : transcription
        # Column Number:        0          | 1          | 2           | 3                | 4              | 5          |
        # Column Description:   GeneSym    | NMID       | Chrom       | Strand           | Txn Start      | Txn End    |
        # Column Example:       TNFSF10    | NM_003810  | chr3        |  -               | 172223297      | 172241297  |

        # Column Number:        6          | 7          | 8           | 9                | 10             |
        # Column Description:   ORF Start  | ORF End    | Exon Count  | Exon Start List  | Exon End Start |
        # Column Example:       172224281  | 172241174  | 5           |  172223297....   | 172224709..... |

        list_sColumn = sReadLine.strip('\n').split('\t')
        cRef         = cRefSeqData()

        # V-S Check: list_sColumn Size
        if len(list_sColumn) < 11:
            sys.exit('ERROR: list_sColumn Size= %d' % len(list_sColumn))

        # Assign each column data as instance variables
        cRef.sGeneSym       = list_sColumn[0].upper()
        cRef.sGeneSymAlt    = ''
        cRef.sNMID          = list_sColumn[1]
        cRef.sNMIDAlt       = ''
        cRef.sChrID         = list_sColumn[2]
        cRef.nChrID         = check_convert_chrID(list_sColumn[2])

        if cRef.nChrID == 0: continue

        cRef.sStrand        = list_sColumn[3]
        cRef.nTxnStartPos   = int(list_sColumn[4])
        cRef.nTxnEndPos     = int(list_sColumn[5])
        cRef.nORFStartPos   = int(list_sColumn[6])
        cRef.nORFEndPos     = int(list_sColumn[7])
        cRef.nExonCount     = int(list_sColumn[8])

        # nExonStart and nExonEnd are comma separated
        sExonStartList      = list_sColumn[9].split(',')[:-1]  # Last element is empty due to final comma on the last position
        cRef.list_nExonS    = [int(StartLoc) for StartLoc in sExonStartList]

        sExonEndList        = list_sColumn[10].split(',')[:-1]  # Last element is empty due to final comma on the last position
        cRef.list_nExonE    = [int(EndLoc) for EndLoc in sExonEndList]

        # V-S Check
        # Exon count and exon start/end position list size
        if (len(cRef.list_nExonS) != len(cRef.list_nExonE)) or \
                (len(cRef.list_nExonS) != cRef.nExonCount) or \
                (len(cRef.list_nExonE) != cRef.nExonCount):
            sys.exit("ERROR: Exon Positions: StartList= %d EndList= %d ExonCount= %d"
                     % (len(cRef.list_nExonS), len(cRef.list_nExonE), cRef.nExonCount))
        # Start positions and end positions
        if cRef.nTxnStartPos > cRef.nTxnEndPos: sys.exit('ERROR: Txn Start= %d : Txn End= %d' % (cRef.nTxnStartPos, cRef.nTxnEndPos))
        if cRef.nORFStartPos > cRef.nORFEndPos: sys.exit('ERROR: ORF Start= %d : ORF End= %d' % (cRef.nORFStartPos, cRef.nORFEndPos))

        obtain_sORFSeqPos(cRef)

        list_sOutput.append(cRef)
    #loop END: sReadLine

    if sOutput == 'dict':

        dict_sOutput = {}
        for cRef in list_sOutput:

            if sKeyType == 'NMID':
                sKey = cRef.sNMID
            else:
                sKey = cRef.sGeneSym

            if sKey not in dict_sOutput:
                dict_sOutput[sKey] = []
            dict_sOutput[sKey].append(cRef)
        #loop END: cRef

        return dict_sOutput

    else:
        return list_sOutput
#def END: parse_refflat_line


def filter_refflat(list_cRef):
    dict_cRef = {}
    for cRef in list_cRef:
        if cRef.sGeneSym not in dict_cRef:
            dict_cRef[cRef.sGeneSym] = []
        dict_cRef[cRef.sGeneSym].append(cRef)
    #loop END: cRef

    list_sOutput = []

    for sGene in dict_cRef:
        list_sNMIDs = [int(cRef.sNMID.replace('NM_', '')) for cRef in dict_cRef[sGene]]

        if len(list_sNMIDs) == 1:
            list_sOutput += dict_cRef[sGene]
        else:
            nMinID = min(list_sNMIDs)

            for cRef in dict_cRef[sGene]:

                sKey = int(cRef.sNMID.replace('NM_', ''))

                if sKey == nMinID: list_sOutput.append(cRef)
            #loop END: cRef
        #if END:
    #loop END: sGene

    dict_sOutput = {}
    for cRef in list_sOutput:

        sKey = cRef.sGeneSym

        if sKey not in dict_sOutput:
            dict_sOutput[sKey] = []
        dict_sOutput[sKey].append(cRef)
    #loop END: cRef

    return dict_sOutput
#def END: filter_refflat


def obtain_sORFSeq(cRef, cGenome):

    cRef.list_nORFStart  = [cRef.nORFStartPos] + [nPos for nPos in cRef.list_nExonS if cRef.nORFStartPos <= nPos < cRef.nORFEndPos]
    cRef.list_nORFEnd    = [nPos for nPos in cRef.list_nExonE if cRef.nORFStartPos <= nPos < cRef.nORFEndPos] + [cRef.nORFEndPos]
    sORFSeq              = ''
    cRef.dict_nExonSeq   = {}


    #if cRef.sStrand == '-':
        #cRef.list_nORFStart = cRef.list_nORFStart[::-1]
        #cRef.list_nORFEnd   = cRef.list_nORFEnd[::-1]

    for i, (nStart, nEnd) in enumerate(zip(cRef.list_nORFStart, cRef.list_nORFEnd)):

        #sSeq     = cGenome.fetch(cRef.sChrID, nStart if cRef.sStrand == '+' else nStart-1, nEnd).upper()
        sSeq     = cGenome.fetch(cRef.sChrID, nStart, nEnd).upper()
        sORFSeq += sSeq

        nExonKey = (nStart+1, nEnd)
        if nExonKey not in cRef.dict_nExonSeq:
            cRef.dict_nExonSeq[nExonKey] = ''
        cRef.dict_nExonSeq[nExonKey] = sSeq

        #print('%s:%s-%s' % (cRef.sChrID, nStart, nEnd))
        #print(sExonKey, sSeq)

    #loop END: nStart, nEnd

    cRef.sORFSeq    = sORFSeq if cRef.sStrand == '+' else reverse_complement(sORFSeq)
#def END: obtain_sORFSeq

def obtain_genomic_pos(cRef, cGenome, sWTSeq, nTarLen, nAltIndex):

    cRef.list_nORFStart  = [cRef.nORFStartPos] + [nPos for nPos in cRef.list_nExonS if cRef.nORFStartPos <= nPos < cRef.nORFEndPos]
    cRef.list_nORFEnd    = [nPos for nPos in cRef.list_nExonE if cRef.nORFStartPos <= nPos < cRef.nORFEndPos] + [cRef.nORFEndPos]
    cRef.dict_nExonSeq   = {}

    sTargetSeq           = sWTSeq[:nTarLen] if cRef.sStrand == '+' else reverse_complement(sWTSeq[:nTarLen])
    nIntronBuffer        = 10

    for i, (nStart, nEnd) in enumerate(zip(cRef.list_nORFStart, cRef.list_nORFEnd)):
        sSeq     = cGenome.fetch(cRef.sChrID, nStart - nIntronBuffer, nEnd + nIntronBuffer).upper()
        nExonKey = (nStart+1, nEnd)
        if nExonKey not in cRef.dict_nExonSeq:
            cRef.dict_nExonSeq[nExonKey] = ''
        cRef.dict_nExonSeq[nExonKey] = sSeq
    #loop END: nStart, nEnd

    list_sTarMatch  = []

    for nExonKey in cRef.dict_nExonSeq:

        nExonS, nExonE = nExonKey
        sExonSeq       = cRef.dict_nExonSeq[nExonKey]

        for sReIndex in regex.finditer(sTargetSeq.upper(), sExonSeq, overlapped=False):
            nIndexStart = sReIndex.start()
            nIndexEnd   = sReIndex.end()
            nGenomicS   = nExonS - nIntronBuffer + nIndexStart
            nGenomicE   = nExonS - nIntronBuffer + nIndexEnd - 1

            sWTSeqCheck = sExonSeq[nIndexStart:nIndexEnd]
            if cRef.sStrand == '+':
                nAltPos_genomic = nGenomicS + nAltIndex
            else:
                nAltPos_genomic = nGenomicS + (nTarLen - nAltIndex) - 1

            #print(nExonKey, sExonSeq)
            #print('%s:%s-%s' % (cRef.sChrID, nGenomicS, nGenomicE), nIndexStart, nIndexEnd, nGenomicE-nGenomicS)
            #print(nAltPos_genomic, sWTSeqCheck)

            list_sTarMatch.append(nAltPos_genomic)
        #loop END: sReIndex
    #loop END: nExonKey

    return list_sTarMatch[0]
#def END: obtain_genomic_pos


def obtain_sORFSeq_UTRs_EditingNeeded(cRef, sChromoSeq):

    list_nExonS     = cRef.list_nExonS
    list_nExonE     = cRef.list_nExonE

    n5UTRStartList  = []
    n5UTREndList    = [cRef.nORFStartPos]  # For '+' Stand, coding start pos is the end

    nORFStartList   = [cRef.nORFStartPos]
    nORFEndList     = [cRef.nORFEndPos]

    n3UTRStartList  = [cRef.nORFEndPos]  # For '-' Stand, coding end pos is the start
    n3UTREndList    = []

    sORFSeq     = ''
    s5UTRSeq    = ''
    s3UTRSeq    = ''

    # Divide up the exon start positions into three lists 5', ORF, and 3'
    for nStartPos in list_nExonS:
        if nStartPos <= cRef.nORFStartPos:
            n5UTRStartList.append(nStartPos)  # Positions before ORF Start Position

        elif cRef.nORFStartPos <= nStartPos < cRef.nORFEndPos:
            nORFStartList.append(nStartPos)  # Positions Between ORF Start and End Position
            nORFStartList = sorted(nORFStartList)

        else:
            n3UTRStartList.append(nStartPos)
            n3UTRStartList = sorted(n3UTRStartList)  # Positions after ORF End Position
    #loop END: nStartPos

    # Divide up the exon end positions into three lists 5', ORF, and 3'
    for nEndPos in list_nExonE:
        if nEndPos <= cRef.nORFStartPos:
            n5UTREndList.append(nEndPos)
            n5UTREndList = sorted(n5UTREndList)

        elif cRef.nORFStartPos <= nEndPos < cRef.nORFEndPos:
            nORFEndList.append(nEndPos)
            nORFEndList = sorted(nORFEndList)

        else:
            n3UTREndList.append(nEndPos)
    #loop END: nEndPos

    # V-S Check, The size of start and end position lists
    if (len(n5UTRStartList) != len(n5UTREndList)) or (len(nORFStartList) != len(nORFEndList)) or (
            len(n3UTRStartList) != len(n3UTREndList)):
        sys.exit(
            'ERROR: Unequal List Sizes:\n 5UTR Start= %d End %d \n ORF Start= %d End= %d \n 3UTR Start= %d End %d' %
            ((len(n5UTRStartList), len(n5UTREndList), len(nORFStartList), len(nORFEndList), len(n3UTRStartList),
              len(n3UTREndList))))
    #if END: V-S Check

    # Compile each segement by range slicing the original chromosome sequence
    for i in range(len(n5UTRStartList)):
        s5UTRSeq = s5UTRSeq + sChromoSeq[n5UTRStartList[i]:n5UTREndList[i]]
    #loop END: i

    for i in range(len(nORFStartList)):
        sORFSeq = sORFSeq + sChromoSeq[nORFStartList[i]:nORFEndList[i]]
        print(i, sORFSeq)

    #loop END: i

    for i in range(len(n3UTRStartList)):
        s3UTRSeq = s3UTRSeq + sChromoSeq[n3UTRStartList[i]:n3UTREndList[i]]
    #loop END: i

    if cRef.sStrand == '-':
        sReverseORFSeq = reverse_complement(sORFSeq)

        # For '-' strand, switch the 5' and the 3'
        sReverse5UTRSeq = reverse_complement(s3UTRSeq)
        sReverse3UTRSeq = reverse_complement(s5UTRSeq)
        cRef.s5UTRSeq   = sReverse5UTRSeq
        cRef.s3UTRSeq   = sReverse3UTRSeq
        cRef.sORFSeq    = sReverseORFSeq
        cRef.n5UTRLen   = len(sReverse5UTRSeq)
        cRef.nORFLen    = len(sReverseORFSeq)
        cRef.n3UTRLen   = len(sReverse3UTRSeq)

    else:  # if Strand is '+'
        cRef.s5UTRSeq   = s5UTRSeq
        cRef.sORFSeq    = sORFSeq
        cRef.s3UTRSeq   = s3UTRSeq
        cRef.n5UTRLen   = len(s5UTRSeq)
        cRef.nORFLen    = len(sORFSeq)
        cRef.n3UTRLen   = len(s3UTRSeq)
    #if END: self.sStrand
#def END: obtain_sORFSeq


def obtain_sORFSeqPos(cRef):

    cRef.n5UTRStartList  = []
    cRef.n5UTREndList    = [cRef.nORFStartPos]  # For '+' Stand, coding start pos is the end

    cRef.nORFStartList   = [cRef.nORFStartPos]
    cRef.nORFEndList     = [cRef.nORFEndPos]

    cRef.n3UTRStartList  = [cRef.nORFEndPos]  # For '-' Stand, coding end pos is the start
    cRef.n3UTREndList    = []

    # Divide up the exon start positions into three lists 5', ORF, and 3'
    for nStartPos in cRef.list_nExonS:
        if nStartPos <= cRef.nORFStartPos:
            cRef.n5UTRStartList.append(nStartPos)  # Positions before ORF Start Position

        elif cRef.nORFStartPos <= nStartPos < cRef.nORFEndPos:
            cRef.nORFStartList.append(nStartPos)  # Positions Between ORF Start and End Position
            cRef.nORFStartList = sorted(cRef.nORFStartList)

        else:
            cRef.n3UTRStartList.append(nStartPos)
            cRef.n3UTRStartList = sorted(cRef.n3UTRStartList)  # Positions after ORF End Position
    #loop END: nStartPos

    # Divide up the exon end positions into three lists 5', ORF, and 3'
    for nEndPos in cRef.list_nExonE:
        if nEndPos <= cRef.nORFStartPos:
            cRef.n5UTREndList.append(nEndPos)
            cRef.n5UTREndList = sorted(cRef.n5UTREndList)

        elif cRef.nORFStartPos <= nEndPos < cRef.nORFEndPos:
            cRef.nORFEndList.append(nEndPos)
            cRef.nORFEndList = sorted(cRef.nORFEndList)

        else:
            cRef.n3UTREndList.append(nEndPos)
    #loop END: nEndPos

    # V-S Check, The size of start and end position lists
    if (len(cRef.n5UTRStartList) != len(cRef.n5UTREndList)) or (len(cRef.nORFStartList) != len(cRef.nORFEndList)) or (
            len(cRef.n3UTRStartList) != len(cRef.n3UTREndList)):
        sys.exit('ERROR: Unequal List Sizes:\n 5UTR Start= %d End %d \n ORF Start= %d End= %d \n 3UTR Start= %d End %d' %
            ((len(cRef.n5UTRStartList), len(cRef.n5UTREndList), len(cRef.nORFStartList), len(cRef.nORFEndList), len(cRef.n3UTRStartList),
              len(cRef.n3UTREndList))))
    #if END: V-S Check
#def END: obtain_sORFSeqPos

##endregion

## region Class cTCGAData

class cTCGAData: pass


## endregion

## region Class cGeneDef
def whereStart(cGene):
    # Function that locates an exon containing cds start.
    nCstart = cGene.nORFSPos

    for nEnum in range(cGene.nNumExons):  # from 0 to len - 1
        if cGene.lExonS[nEnum] <= nCstart and nCstart < cGene.lExonE[nEnum]:
            return nEnum
    # for nEnum end.


# whereStart end.


def whereEnd(cGene):
    # Function that locates an exon containg cds end.
    nCend = cGene.nORFEPos

    for nEnum in range(cGene.nNumExons):  # from 0 to len - 1
        if cGene.lExonS[nEnum] < nCend and nCend <= cGene.lExonE[nEnum]:
            return nEnum
    # for nEnum end.


# whereEnd end.


def lengthUTR(cGene):
    # Function that returns the length of 3'UTR and 5'UTR
    sSign = cGene.sStrand
    nUTR3 = 0
    nUTR5 = 0

    nFrom = whereStart(cGene)
    nTill = whereEnd(cGene)
    nUTR3 += cGene.lExonE[nTill] - cGene.nORFEPos
    nUTR5 += cGene.nORFSPos - cGene.lExonS[nFrom]

    while nTill < cGene.nNumExons - 1:
        nTill += 1
        nUTR3 += cGene.lExonE[nTill] - cGene.lExonS[nTill]
    # while nTill end.

    while 0 < nFrom:
        nFrom -= 1
        nUTR5 += cGene.lExonE[nFrom] - cGene.lExonS[nFrom]
    # while i end.

    if sSign == '-':
        nTemp = nUTR3
        nUTR3 = nUTR5
        nUTR5 = nTemp
    # if end.

    return (nUTR5, nUTR3)


# LengthUTR end.


def readSeq(nChr, nStart, nEnd, sChrSeqAddress):
    # Function that returns the sequence.
    if nChr == 23:
        sChr = 'X'
    elif nChr == 24:
        sChr = 'Y'
    else:
        sChr = str(nChr)

    file = open('%s/chr%s.fa' % (sChrSeqAddress, sChr))
    file.seek(nStart)
    sSeq = file.read(nEnd - nStart)
    file.close()
    return sSeq


# readSeq end.


def revSeq(sSeq):
    sRseq = sSeq[::-1]

    sRseq = sRseq.replace('A', 't')
    sRseq = sRseq.replace('T', 'A')
    sRseq = sRseq.replace('t', 'T')
    sRseq = sRseq.replace('G', 'c')
    sRseq = sRseq.replace('C', 'G')
    sRseq = sRseq.replace('c', 'C')
    return sRseq


# revSeq end.


def readExons(cGene, sChrSeqAddress):
    sSeq = ''
    for i in range(cGene.nNumExons):
        sSeq += readSeq(cGene.nChrID, cGene.lExonS[i], cGene.lExonE[i], sChrSeqAddress)
    # for i end.
    return sSeq


# readExons end.


class cGeneDef:
    '''
        @INSTANCE VARIABLES
        sGeneSym :      gene symbol ex) 'TP53'
        sNMID :         NMID, as a string ex) 'NM_123'
        sStrand :       strand, as a string ex) '+'
        nChrID :        chromosome ID, as a string ex) '22' for chr22
        nTransSPos :    transcription start position, 0-based coordinate(python-like)
        nTransEPos :    transcription end position, 0-based coordinate(python-like)
        nORFSPos :      Open reading frame(ORF) start position
        nORFEPos :      ORF end position
        nNumExons :     Number of exons
        lExonS :        list of exon start positions
        lExonE :        list of exon end positions
        lORFExonS :     list of ORF start positions
        lORFExonE :     list of ORF end positions
        nExonSeqSize :  the number of whole exons(=transcript)
        n3UTRSize :     the number of characters in 3'UTR
        n5UTRSize :     the number of characters in 5'UTR
        nORFSize :      the number of characters in ORF
        sExonSeq :      sequence of whole exons (=transcript)
        s5UTRSeq :      sequence of 5'UTR
        s3UTRSeq :      sequence of 3'UTR
        nReads :        the number of reads mapped (into whole Tx, depends on the running code)
        nIntronReads :  the number of reads mapped into the gene's intronic region (not 100% accurate)
        n3UTRReads :    the number of reads mapped into the gene's 3'UTR
        nORFReads :     the number of reads mapped into the gene's ORF
        n5UTRReads :    the number of reads mapped into the gene's 5'UTR


        @methods
        __init__() :    initialize
        readInputLine(sRefFlatLine) :       parse gene information from a RefFlat line
        seqInfo :                           get sequence information
    '''

    def __init__(self):
        self.sGeneSym = 'NULL'
        self.sNMID = 'NULL'
        self.sStrand = 'NULL'
        self.nNMID = -1
        self.nChrID = 0
        self.nTransSPos = 0
        self.nTransEPos = 0
        self.nORFSPos = 0
        self.nORFEPos = 0
        self.nNumExons = 0
        self.lExonS = []
        self.lExonE = []

        self.nReads = 0
        self.nIntronReads = 0
        self.n3UTRReads = 0
        self.nORFReads = 0
        self.n5UTRReads = 0

    # __init__() end.

    def readInputLine(self, sReadLine):
        lReadLine = sReadLine.strip().split('\t')

        self.sGeneSym = lReadLine[0]
        self.sNMID = lReadLine[1]
        self.sStrand = lReadLine[3]

        self.nTransSPos = int(lReadLine[4])
        self.nTransEPos = int(lReadLine[5])
        self.nORFSPos = int(lReadLine[6])
        self.nORFEPos = int(lReadLine[7])
        self.nNumExons = int(lReadLine[8])

        self.lExonS = [int(x) for x in lReadLine[9].split(',') if x != '']
        self.lExonE = [int(x) for x in lReadLine[10].split(',') if x != '']

        self.lORFExonS = [nExonS for nExonS in self.lExonS if nExonS > self.nORFSPos and nExonS < self.nORFEPos]
        self.lORFExonE = [nExonE for nExonE in self.lExonE if nExonE > self.nORFSPos and nExonE < self.nORFEPos]
        self.lORFExonS.insert(0, self.nORFSPos)
        self.lORFExonE.append(self.nORFEPos)

        self.nChrID = lReadLine[2][3:]

    # ReadInputLine() end.

    def seqInfo(self, sChrSeqAddress):
        sSeq = readExons(self, sChrSeqAddress)
        (nUTR5, nUTR3) = lengthUTR(self)

        if self.sStrand == '-':
            sSeq = revSeq(sSeq)

        self.nExonSeqSize = len(sSeq)
        self.n3UTRSize = nUTR3
        self.n5UTRSize = nUTR5
        self.nORFSize = len(sSeq) - nUTR3 - nUTR5
        self.sExonSeq = sSeq

        self.s5UTRSeq = sSeq[:nUTR5]
        if nUTR3 != 0:
            self.sORFSeq = sSeq[nUTR5: - nUTR3]
            self.s3UTRSeq = sSeq[-nUTR3:]
        else:
            self.sORFSeq = sSeq[nUTR5:]
            self.s3UTRSeq = ''
    # SeqInfo end.


# Class cGene() end.


# This function is for genePred format(RefFlat)
def generateGeneList(sFileAdress):
    lGeneList = []
    with open(sFileAdress) as file:
        for sEachLine in file:
            lGeneList.append(cGeneDef())
            lGeneList[-1].readInputLine(sEachLine)
        # for end.
    # with end.
    return lGeneList


# GenerateGeneList() end.


# This function is for gtf file format
def generateGeneList_GTF(sFileAddress):
    lGeneList = []

    with open(sFileAddress) as fileG:
        lGTF = [sLine.strip().split(None, 8) for sLine in fileG.readlines()]
    # with end.

    dmRNA = {}
    dExon = {}
    dCDS = {}

    for lLine in lGTF:
        lTemp = lLine[8].split(';')

        if lLine[2] == 'mRNA':
            sTxID = lTemp[0].split('=')[1]
            # sTxID must be unique since we selected representative isoforms
            dmRNA[sTxID] = lLine

        elif lLine[2] == 'exon':
            sTxID = lTemp[1].split('=')[1]
            if sTxID in dExon:
                dExon[sTxID].append(lLine)
            else:
                dExon[sTxID] = [lLine]
            # if end.
        elif lLine[2] == 'CDS':
            sTxID = lTemp[2].split('=')[1]
            if sTxID in dCDS:
                dCDS[sTxID].append(lLine)
            else:
                dCDS[sTxID] = [lLine]
            # if end.
        # if end.

    # for lLine end.

    for sTxID in dmRNA:
        cGene = cGeneDef()

        ''' From 'mRNA' line '''
        lmRNALine = dmRNA[sTxID]
        lmRNATemp = lmRNALine[8].split(';')
        cGene.sGeneSym = [sItem for sItem in lmRNATemp if sItem.startswith('gene=')][0].split('=')[1]
        cGene.sNMID = lmRNATemp[1].split('=')[1]
        cGene.sStrand = lmRNALine[6]
        cGene.nChrID = lmRNALine[0][3:]
        cGene.nTransSPos = int(lmRNALine[3]) - 1  # converting into 0-based index
        cGene.nTransEPos = int(lmRNALine[4])
        cGene.sRNAID = sTxID

        ''' From 'exon' lines '''
        lExonLines = dExon[sTxID]

        cGene.lExonS = sorted([int(lExon[3]) - 1 for lExon in lExonLines])
        cGene.lExonE = sorted([int(lExon[4]) for lExon in lExonLines])
        cGene.nNumExons = len(lExonLines)

        ''' From 'CDS' lines '''
        if sTxID in dCDS:
            lCDSLines = dCDS[sTxID]

            cGene.lORFExonS = sorted([int(lCDS[3]) - 1 for lCDS in lCDSLines])
            cGene.lORFExonE = sorted([int(lCDS[4]) for lCDS in lCDSLines])
            cGene.nORFSPos = cGene.lORFExonS[0]
            cGene.nORFEPos = cGene.lORFExonE[-1]
            lGeneList.append(cGene)
        else:
            # rna8497 is only one which belongs to this case. weired.
            # So exclude from the gene list.
            cGene.lORFExonS = [0]
            cGene.lORFExonE = [0]
        # if end.

    # for sTxID end.

    return lGeneList


# generateGeneList_GTF() end.


def addSeqData(lGeneList, sChrDir):
    for cGene in lGeneList:
        cGene.seqInfo(sChrDir)
    # for cGene end.

    return lGeneList


# addSeqData() end.


## endregion

## region class cCosmic
class cCOSMIC:
    def __init__(self):
        self.sGeneName = ''
        self.sAccID = ''
        self.nCDSLen = 0
        self.sHGCNID = ''  # SKIP for now
        self.sSample = ''  # SKIP for now
        self.sSampleID = ''  # SKIP for now
        self.sTumorID = ''  # SKIP for now
        self.sPriSite = ''  # primary site  ex) pancreas
        self.sSiteSub1 = ''  # SKIP for now
        self.sSiteSub2 = ''  # SKIP for now
        self.sSiteSub3 = ''  # SKIP for now
        self.sPriHist = ''  # primary histology
        self.sHistSub1 = ''  # SKIP for now
        self.sHistSub2 = ''  # SKIP for now
        self.sHistSub3 = ''  # SKIP for now
        self.bGenomeWide = ''  # ex) y or n
        self.sMutaID = ''  # SKIP for now
        self.sAltType = ''  # ex) c.35G>T
        self.sRef = ''
        self.sAlt = ''
        self.sAAType = ''  # ex) p.G12V
        self.sMutaDescri = ''  # ex) Substitution - Missense
        self.sMutaZygo = ''  # SKIP for now
        self.bLOH = ''  # loss of heterzygosity ex) y or n
        self.sGRCh = ''  # Genome Version
        self.sGenicPos = ''  # 17:7673781-7673781
        self.nChrID = ''  # 17  X = 24 Y = 25
        self.sChrID = ''  # chr17 or chrX and chrY
        self.sPos = ''  # 7673781   1-based
        self.sStrand = ''
        self.bSNP = ''  # ex) y and n
        self.sDelete = ''  # ex) PATHOGENIC
    # def END : __init__


def cCos_parse_cosmic_consensus(sWorkDir, sCosmicFile):
    dict_sOutput = {}
    InFile = open(sCosmicFile, 'r')

    for i, sReadLine in enumerate(InFile):

        if sReadLine.startswith('Gene'): continue

        list_sColumn = sReadLine.strip('\n').split('\t')
        '''
        if i == 0:
            list_sHeader = list_sColumn
        elif i == 1:
            for i,(a,b) in enumerate(zip(list_sHeader, list_sColumn)):
                print('%s\t%s\t%s' % (i,a,b))
        else: break

        '''
        cCos             = cCOSMIC()
        cCos.sGeneName   = list_sColumn[0].upper()
        cCos.sAccID      = list_sColumn[1]
        cCos.nCDSLen     = int(list_sColumn[2])
        cCos.sHGCNID     = list_sColumn[3]
        cCos.sSample     = list_sColumn[4]
        cCos.sSampleID   = list_sColumn[5]
        cCos.sTumorID    = list_sColumn[6]
        cCos.sPriSite    = list_sColumn[7]
        cCos.sSiteSub1   = list_sColumn[8]
        cCos.sSiteSub2   = list_sColumn[9]
        cCos.sSiteSub3   = list_sColumn[10]
        cCos.sPriHist    = list_sColumn[11]
        cCos.sHistSub1   = list_sColumn[12]
        cCos.sHistSub2   = list_sColumn[13]
        cCos.sHistSub3   = list_sColumn[14]
        cCos.bGenomeWide = True if list_sColumn[15] == 'y' else False
        cCos.sMutaID     = list_sColumn[16]
        cCos.sAltType    = list_sColumn[17]
        cCos.sAAType     = list_sColumn[18]
        cCos.sMutaDescri = list_sColumn[19]
        cCos.sMutaZygo   = list_sColumn[20]
        cCos.bLOH        = True if list_sColumn[21] == 'y' else False
        cCos.sGRCh       = list_sColumn[22]
        cCos.sGenicPos   = list_sColumn[23]
        if not list_sColumn[23]: continue  # Skip those w/o position information

        cCos.nChrID = list_sColumn[23].split(':')[0]

        if cCos.nChrID not in ['24', '25']:
            cCos.sChrID = 'chr%s' % cCos.nChrID
        else:
            dict_sChrKey = {'24': 'chrX', '25': 'chrY'}
            cCos.sChrID  = dict_sChrKey[cCos.nChrID]
        # if END

        list_sPosCheck   = list(set(list_sColumn[23].split(':')[1].split('-')))

        if len(list_sPosCheck) > 1:
            cCos.sPos = list_sPosCheck[0]
        else:
            cCos.sPos = ''.join(list_sPosCheck)
        #if END:

        cCos.sStrand = list_sColumn[24]
        cCos.bSNP = True if list_sColumn[25] == 'y' else False

        if cCos.sChrID not in dict_sOutput:
            dict_sOutput[cCos.sChrID] = []
        dict_sOutput[cCos.sChrID].append(cCos)
    #loop END: i, sReadLine
    InFile.close()
    # V-S Check:
    if not dict_sOutput:
        sys.exit('Empty List : cCos_parse_cosmic_consensus : dict_sOutput : Size = %d' % (len(dict_sOutput)))

    for sChrID in dict_sOutput:
        sPickle     = '%s/list_cCos_%s.pickle' % (sWorkDir, sChrID)
        list_cCos   = dict_sOutput[sChrID]

        print(sChrID, len(list_cCos))

        PickleOut   = open(sPickle, 'wb')
        pickle.dump(list_cCos, PickleOut)
        PickleOut.close()
    #loop END: sChrID
#def END: cCos_parse_cosmic_consensus

#class END: cCosmic

## endregion


##endregion

## region Util Functions

def copy_temp_core_script(sWorkDir):
    os.makedirs('%s/temp' % sWorkDir, exist_ok=True)
    os.system('cp %s/B_K1000E_additional_anals.py %s/temp/tmp_script_%s.py'
              % (sWorkDir, sWorkDir, sTIME_STAMP))
    return '%s/temp/tmp_script_%s.py' % (sWorkDir, sTIME_STAMP)
#def END: copy_temp_core_script


def reverse_complement(sSeq):
    dict_sBases = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'N': 'N', '.': '.', '*': '*',
                   'a': 't', 'c': 'g', 'g': 'c', 't': 'a'}
    list_sSeq   = list(sSeq)  # Turns the sequence in to a gigantic list
    list_sSeq   = [dict_sBases[sBase] for sBase in list_sSeq]
    return ''.join(list_sSeq)[::-1]
#def END: reverse_complement


def check_convert_chrID(sChrID):
    sChrID = sChrID.replace('chr', '').upper()  # remove chr- prefix

    if sGENOME in ['hg19', 'hg38']:
        if sChrID in list_sCHRIDs_HG:  # Only [1-22, X and Y]
            if sChrID == 'X':
                return 23
            elif sChrID == 'Y':
                return 24
            elif sChrID == 'M':
                return 0
            else:
                return int(sChrID)
        else:
            return 0  # Case Example: chr6_qbl_hap6

    elif sGENOME == 'mm10':
        if sChrID in list_sCHRIDs_MM:  # Only [1-22, X and Y]
            if sChrID == 'X':
                return 20
            elif sChrID == 'Y':
                return 21
            elif sChrID == 'M':
                return 0
            else:
                return int(sChrID)
        else:
            return 0  # Case Example: chr6_qbl_hap6
#def END: process_chromo_ID


def load_extendfile(sInFile):
    InFile       = open(sInFile, 'r')
    list_sOutput = []

    for sReadLine in InFile:
        list_sColumn = sReadLine.strip('\n').split('\t')

        list_sOutput.append(list_sColumn)
    #loop END: sReadLine
    return list_sOutput
#def END: load_extendfile


def parse_genelist(sInFile):
    dict_sOutput    = {}
    InFile          = open(sInFile, 'r')

    for sReadLine in InFile:
        sGeneSym    = sReadLine.replace('\n', '').split('\t')[0].upper()
        sNMID       = sReadLine.replace('\n', '').split('\t')[1].split('.')[0].upper()

        if sGeneSym not in dict_sOutput:
            dict_sOutput[sGeneSym] = []
        dict_sOutput[sGeneSym].append(sNMID)
    #loop END: sReadLine

    list_sOutput = [[sGeneSym, list(set(dict_sOutput[sGeneSym]))[0]] for sGeneSym in dict_sOutput]

    return list_sOutput
#def END: parse_genelist


def get_gene_info(sWorkDir, list_sTargetGenes):
    sOutDir     = '%s/OutputPerGene' % sWorkDir
    os.makedirs(sOutDir, exist_ok=True)

    list_sKeys  = ['cellline', 'chr', 'start', 'end', 'symbol', 'sequence', 'strand', 'pubmed',
                  'cas', 'screentype', 'condition', 'effect', 'ensg', 'log2fc']

    sHeader     = '\t'.join(list_sKeys)

    for sGene in list_sTargetGenes:

        print('Processing %s' % sGene)

        sOutFile    = '%s/%s.output.txt' % (sOutDir, sGene)
        OutFile     = open(sOutFile, 'w')
        OutFile.write('%s\n' % sHeader)

        sScript     = 'curl -s -H "Content-Type: application/json" '
        sScript     += '-X POST -d \'{"query":"%s"}\' ' % sGene
        sScript     += 'http://genomecrispr.dkfz.de/api/sgrnas/symbol; '

        sStdOut     = subprocess.Popen(sScript, stdout=subprocess.PIPE, shell=True).stdout
        list_sJSON  = json.load(sStdOut)

        for dict_sInfo in list_sJSON:
            list_sInfo  = ['%s' % dict_sInfo[sKey] for sKey in list_sKeys]

            sOut        = '\t'.join(list_sInfo)
            OutFile.write('%s\n' % sOut)
        #loop END: dict_sInfo
        OutFile.close()

        print('Processing %s.............Done' % sGene)
    #loop END: sGene
#def END: get_gene_info


def get_lines_wTargetGenes(list_sTargetGenes, sGeneInfoFile, sOutputFile):
    if not os.path.isfile(sGeneInfoFile): sys.exit('File Not Found %s' % sGeneInfoFile)

    InFile  = open(sGeneInfoFile, 'r')
    OutFile = open(sOutputFile, 'w')

    for sReadLine in InFile:

        if sReadLine.startswith('start'): continue

        list_sColumn    = sReadLine.replace('\n', '').split(',')
        sGeneSym        = list_sColumn[8]

        if sGeneSym in list_sTargetGenes: OutFile.write(sReadLine)
    #loop END: sReadLine
    InFile.close()
    OutFile.close()
#def END: get_lines_wTargetGenes


def matplot():
    print('MATPLOTLIB - Symplot')
    sInFile         = '%s/input/matplot_data.txt' % sDATA_DIR
    list_X, list_Y  = load_matplot_data(sInFile)

    assert len(list_X) == len(list_Y)

    sOutDir         = '%s/output/matplot' % sDATA_DIR
    sOutFile        = '%s/test_xylinthresh5axislimited300.png' % sOutDir

    ### Figure Size ###
    FigWidth        = 10
    FigHeight        = 10

    OutFig = plt.figure(figsize=(FigWidth, FigHeight))
    SubPlot = OutFig.add_subplot(111)

    ### Marker ###########
    Red             = 0
    Green           = 0
    Blue            = 0
    MarkerSize      = 50
    Circle          = 'o'
    DownTriangle    = 'v'
    #######################

    ### Log Start Point ###
    LimitThresh     = 10
    #######################

    ### Axis Range ########
    Xmin = 0
    Xmax = 100
    Ymin = 0
    Ymax = 100
    ########################

    ### Tick Marks #########
    TickScale     = 1
    MajorTickSize = 10
    MinorTickSize = 5

    plt.xlim(xmin=Xmin, xmax=Xmax)
    plt.ylim(ymin=Ymin, ymax=Ymax)
    plt.xscale('symlog', linthreshx=LimitThresh)
    plt.yscale('symlog', linthreshy=LimitThresh)

    plt.axes().xaxis.set_minor_locator(MinorSymLogLocator(TickScale))
    plt.axes().yaxis.set_minor_locator(MinorSymLogLocator(TickScale))

    plt.tick_params(which='major', length=MajorTickSize)
    plt.tick_params(which='minor', length=MinorTickSize)

    ScaledRed   = Red / 255
    ScaledGreen = Green / 255
    ScaledBlue  = Blue / 255

    SubPlot.scatter(list_X, list_Y, c=[ScaledRed, ScaledGreen, ScaledBlue], marker=Circle, s=MarkerSize)

    OutFig.savefig(sOutFile)
#def END: matplot


def load_matplot_data(sInFile):
    list_X = []
    list_Y = []

    InFile = open(sInFile, 'r')
    for sReadLine in InFile:
        list_sColumns = sReadLine.replace('\n', '').split('\t')

        fXval, fYval  = [float(fValue) for fValue in list_sColumns]

        list_X.append(fXval)
        list_Y.append(fYval)

    #loop END: sReadLine
    return list_X, list_Y
#def END: load_matplot_data


def get_chr_list(sGenome):
    return [line.strip('\n').split('\t')[0][3:] for line in open('%s/%s.fa.fai' % (sGENOME_DIR, sGENOME), 'r')]
#def END: get_chr_list


## region Make Non-redundant RefFlat and GTP File

def generate_filtered_refflat():
    sGenome     = 'mm10'
    sGenomeFile = '%s/%s/%s_refFlat_full.txt'

    sInFile     = '%s/%s/%s_refFlat_full.txt' % (sREF_DIR, sGenome, sGenome)
    sOutFile    = '%s/%s/%s_refFlat_filtered.txt' % (sREF_DIR, sGenome, sGenome)

    lChr        = get_chr_list(sGENOME)  # Get List of Based on .fai file

    # 1 : Load RefFlat file
    lGeneList   = generateGeneList(sInFile)
    countEntry('StartList', lGeneList)

    # 2 : Remove non-NM seq
    lGeneList   = step2(lGeneList)
    countEntry('Step2', lGeneList)

    # 3 : Keep only genes on chr1-22, X, Y
    lGeneList   = step3(lGeneList, lChr)
    countEntry('Step3', lGeneList)

    # 4 : Remove multiple NMID
    lGeneList   = step4(lGeneList)
    countEntry('Step4', lGeneList)

    step7(lGeneList, sOutFile)

    # 5 : Remove genes with wrong ORF
    lGeneList = addSeqData(lGeneList, sCHRSEQ_DIR)
    lGeneList = step5(lGeneList)
    countEntry('Step5', lGeneList)

    # 6 : Remove NMD candidates
    lGeneList = step6(lGeneList, sCHRSEQ_DIR)
    countEntry('Step6', lGeneList)

    # 7 : Output as a new RefFlat file
    step7(lGeneList, sOutFile)

    # step7_GTF(lGeneList, sOutFile_gtf)
#def END: generate_filtered_refflat


def countEntry(sStep, lGeneList):
    print("The number of entries %s : %s" % (sStep, len(lGeneList)))
#def END: countEntry


def step2(lGeneList):
    lNew = []
    for cGene in lGeneList:

        if 'NM' in cGene.sNMID:
            lNew.append(cGene)
        else:
            pass
        # if end.
    # for cGene end.

    return lNew
#def END: step2


def step3(lGeneList, lChr):
    lNew = []

    for cGene in lGeneList:
        if cGene.nChrID in lChr:
            lNew.append(cGene)
        else:
            pass
        # if end.
    # for cGene end.

    return lNew
#def END: step3


def step4(lGeneList):
    lNew = []
    lNMID = [cGene.sNMID for cGene in lGeneList]

    for cGene in lGeneList:
        nNMCount = lNMID.count(cGene.sNMID)
        if nNMCount == 1:
            lNew.append(cGene)
        else:
            pass
        # if end.
    # for cGene end.

    return lNew
#def END: step4


def step5(lGeneList):
    lNew = []

    for cGene in lGeneList:
        sORFSeq = cGene.sORFSeq

        ##Does it has a proper start codon?
        if sORFSeq.startswith('ATG'):
            pass
        else:
            continue
        # if end.

        ##Does it has a proper Stop codon?
        if sORFSeq.endswith('TAA'):
            pass
        elif sORFSeq.endswith('TAG'):
            pass
        elif sORFSeq.endswith('TGA'):
            pass
        else:
            continue
        # if end.

        ##Does it has 3N base pairs?
        if len(sORFSeq) % 3 == 0:
            pass
        else:
            continue
        # if end.

        ##Doesn't it has any premature stop codon?
        lCodon = [sORFSeq[i * 3:(i + 1) * 3] for i in range(int(len(sORFSeq) / 3 - 1))]

        if ('TAA' in lCodon) or ('TAG' in lCodon) or ('TGA' in lCodon):
            continue
        else:
            pass
        # if end.

        lNew.append(cGene)

    # for cGene end.

    return lNew
#def END: step5


def markExon(cGene, sChrSeq):
    lEstarts = cGene.lExonS
    lEends = cGene.lExonE
    nCstart = cGene.nORFSPos
    nCend = cGene.nORFEPos
    nChr = cGene.nChrID
    nFrom = whereStart(cGene)
    nTill = whereEnd(cGene)
    sSeq = ''

    for i in range(len(lEends)):

        if i == nFrom:
            if nFrom == nTill:
                sSeq += readSeq(nChr, lEstarts[i], nCstart, sChrSeq)
                sSeq += '*'
                sSeq += readSeq(nChr, nCstart + 1, nCend - 1, sChrSeq)
                sSeq += '*'
                sSeq += readSeq(nChr, nCend, lEends[i], sChrSeq)
            else:
                sSeq += readSeq(nChr, lEstarts[i], nCstart, sChrSeq)
                sSeq += '*'
                sSeq += readSeq(nChr, nCstart + 1, lEends[i], sChrSeq)
            # fi is cds in the same exon?

        elif i == nTill:
            sSeq += readSeq(nChr, lEstarts[i], nCend - 1, sChrSeq)
            sSeq += '*'
            sSeq += readSeq(nChr, nCend, lEends[i], sChrSeq)
        else:
            sSeq += readSeq(nChr, lEstarts[i], lEends[i], sChrSeq)
        # fi end.

        if i == len(lEends) - 1:
            continue
        else:
            sSeq += '/'
            continue
        # fi put '/' on the end of exon but not on the last one.

    # for i end.
    return sSeq
#def END: markExon


def step6(lGeneList, sChrDir):
    lNew = []
    for cGene in lGeneList:
        nTill = whereEnd(cGene)
        sSeq = markExon(cGene, sChrDir)

        if cGene.sStrand == '+':
            sSeq = sSeq[::-1]
        else:
            pass

        if cGene.sStrand == '+':
            nCdE_from_Last_Exon = len(cGene.lExonE) - 1 - nTill
        else:
            nCdE_from_Last_Exon = whereStart(cGene)
        # nCdE_from_Last_Exon shows that how many exon junctions are located between CdE and LEEJ.
        # nCdE_from_Last_Exon = 0 if it's in the last exon(when considering +/-).

        nCds = sSeq.find('*')
        nEej = sSeq.find('/')

        if nEej == -1:  # no exon-exon junction = no NMD
            lNew.append(cGene)
            continue
        # if end.

        if nCdE_from_Last_Exon > 1:
            nCds -= nCdE_from_Last_Exon - 1
        else:
            pass

        if nCds - nEej > 51:  # NMD candidate
            continue
        else:
            lNew.append(cGene)
        # if end.
    # for cGene end.

    return lNew
#def END: step6


def step7(lGeneList, sOutputRef):
    with open(sOutputRef, 'w') as fileO:

        for cGene in lGeneList:

            fileO.write(cGene.sGeneSym)
            fileO.write('\t')
            fileO.write(cGene.sNMID)
            fileO.write('\t')
            fileO.write('chr' + cGene.nChrID)
            fileO.write('\t')
            fileO.write(cGene.sStrand)
            fileO.write('\t')
            fileO.write(str(cGene.nTransSPos))
            fileO.write('\t')
            fileO.write(str(cGene.nTransEPos))
            fileO.write('\t')
            fileO.write(str(cGene.nORFSPos))
            fileO.write('\t')
            fileO.write(str(cGene.nORFEPos))
            fileO.write('\t')
            fileO.write(str(cGene.nNumExons))
            fileO.write('\t')

            for nNum in cGene.lExonS:
                fileO.write(str(nNum) + ',')
            # for nNum end.

            fileO.write('\t')

            for nNum in cGene.lExonE:
                fileO.write(str(nNum) + ',')
            # for nNum end.

            fileO.write('\n')
        # for cGene end.
    # with end.
#def END: step7


def step7_GTF(lGeneList, sGTFOut):
    nRNAID = 1
    nExonID = 1

    with open(sGTFOut, 'w') as fileO:
        for cGene in lGeneList:
            ''' mRNA Line '''
            # ID=rna4;Name=NM_00;Parent=gene3;;gene=VAMP7;;transcript_id=NM_00
            sTemp = ''
            sTemp += 'ID=rna' + str(nRNAID) + ';'
            sTemp += 'Name=' + cGene.sNMID + ';'
            sTemp += 'Parent=gene' + str(nRNAID) + ';'
            sTemp += 'gene=' + cGene.sGeneSym + ';'
            sTemp += 'transcript_id=' + cGene.sNMID

            fileO.write('chr' + cGene.nChrID)
            fileO.write('\t')
            fileO.write('RefSeq')
            fileO.write('\t')
            fileO.write('mRNA')
            fileO.write('\t')
            fileO.write(str(cGene.nTransSPos + 1))
            fileO.write('\t')
            fileO.write(str(cGene.nTransEPos))
            fileO.write('\t')
            fileO.write('.')
            fileO.write('\t')
            fileO.write(cGene.sStrand)
            fileO.write('\t')
            fileO.write('.')
            fileO.write('\t')
            fileO.write(sTemp)
            fileO.write('\n')

            # Exon Line
            # ID=id4;Parent=rna4;gene=VAMP7;transcript_id=NM_00
            for i in range(cGene.nNumExons):
                sTemp = ''
                sTemp += 'ID=id' + str(nExonID) + ';'
                sTemp += 'Parent=rna' + str(nRNAID) + ';'
                sTemp += 'gene=' + cGene.sGeneSym + ';'
                sTemp += 'transcript_id=' + cGene.sNMID

                fileO.write('chr' + cGene.nChrID)
                fileO.write('\t')
                fileO.write('RefSeq')
                fileO.write('\t')
                fileO.write('exon')
                fileO.write('\t')
                fileO.write(str(cGene.lExonS[i] + 1))
                fileO.write('\t')
                fileO.write(str(cGene.lExonE[i]))
                fileO.write('\t')
                fileO.write('.')
                fileO.write('\t')
                fileO.write(cGene.sStrand)
                fileO.write('\t')
                fileO.write('.')
                fileO.write('\t')
                fileO.write(sTemp)
                fileO.write('\n')
                nExonID += 1
            # for i end.

            nRNAID += 1
        # for cGene end.
    # with end.
#def END: step7_GTF

## endregion


def make_bedfile_for_UCSC ():
    sCosmicFile = '%s/COSMIC/cosmic_mutations_hg38.tsv' % sREF_DIR
    InFile = open(sCosmicFile, 'r')

    dict_sOutput = {}
    for sReadLine in InFile:
        if sReadLine.startswith('Gene'): continue

        list_sColumn = sReadLine.strip('\n').split('\t')

        sGenicPos    = list_sColumn[23]
        nChrID       = list_sColumn[23].split(':')[0]

        if nChrID not in ['24', '25']:
            sChrID = 'chr%s' % nChrID
        else:
            dict_sChrKey = {'24': 'chrX', '25': 'chrY'}
            sChrID  = dict_sChrKey[nChrID]
        # if END
        if sChrID not in dict_sOutput:
            dict_sOutput[sChrID] = {}

        if sGenicPos not in dict_sOutput[sChrID]:
            dict_sOutput[sChrID][sGenicPos] = 0
        dict_sOutput[sChrID][sGenicPos] += 1
    # loop END: sReadLine

    sOutDir  = '%s/COSMIC/forLiftOver' % sREF_DIR
    os.makedirs(sOutDir, exist_ok=True)
    sOutFile = '%s/cosmic_hg38_coords.input.bed' % sOutDir
    OutFile  = open(sOutFile, 'w')

    for sChrID in dict_sOutput:

        for sCoordinate in dict_sOutput[sChrID]:

            if not sCoordinate: continue

            S, E = sCoordinate.split(':')[1].split('-')

            OutFile.write('%s %s %s %s\n' % (sChrID, S, E, sCoordinate))
        #loop END: sCoordinate
    #loop END: sChrID
    OutFile.close()
#def END: make_bedfile_for_UCSC

## endregion

def basic_stat_ABSOLUTE_data (sWorkDir):

    list_sTargetCL = ['DLD1', 'HCT116', 'A375',
                      'A549', 'U2OS', 'K562',
                      'ZR751', 'HS578T',
                      ]

    sInFile        = '%s/CCLE_ABSOLUTE_combined_20181227.txt' % sWorkDir
    InFile         = open(sInFile, 'r')
    list_sOutput   = []

    for sReadLine in InFile:
        ## File Format ##
        #sample	                22RV1_PROSTATE
        #Chromosome	            1
        #Start	                564621
        #End	                111374093
        #Num_Probes	            32625        *number of SNPs detected on segment
        #Length	                110809472
        #Modal_HSCN_1	        1
        #Modal_HSCN_2	        1
        #Modal_Total_CN	        2
        #Subclonal_HSCN_a1	    0
        #Subclonal_HSCN_a2	    0
        #Cancer_cell_frac_a1	0.02
        #Ccf_ci95_low_a1	    0
        #Ccf_ci95_high_a1	    0.03451
        #Cancer_cell_frac_a2	0.02
        #Ccf_ci95_low_a2	    0
        #Ccf_ci95_high_a2	    0.03451
        #LOH	                0
        #Homozygous_deletion	0
        #depMapID	            ACH-000956
        if sReadLine.startswith('sample'): continue

        list_sColumn                = sReadLine.strip('\n').split('\t')
        cCN                         = cAbsoluteCNV()
        cCN.sSample                 = list_sColumn[0]
        cCN.sChrom                  = list_sColumn[1]
        cCN.nStartPos               = int(list_sColumn[2])
        cCN.nEndPos                 = int(list_sColumn[3])
        cCN.nProbCnt                = int(list_sColumn[4])
        cCN.nLen                    = int(list_sColumn[5])
        cCN.Modal_HSCN_1	        = int(list_sColumn[6])
        cCN.Modal_HSCN_2	        = int(list_sColumn[7])
        cCN.Modal_Total_CN	        = int(list_sColumn[8])
        cCN.Subclonal_HSCN_a1	    = int(list_sColumn[9])
        cCN.Subclonal_HSCN_a2	    = int(list_sColumn[10])
        cCN.Cancer_cell_frac_a1	    = float(list_sColumn[11]) if list_sColumn[11] != 'NA' else 'NA'
        cCN.Ccf_ci95_low_a1	        = float(list_sColumn[12]) if list_sColumn[12] != 'NA' else 'NA'
        cCN.Ccf_ci95_high_a1	    = float(list_sColumn[13]) if list_sColumn[13] != 'NA' else 'NA'
        cCN.Cancer_cell_frac_a2	    = float(list_sColumn[14]) if list_sColumn[14] != 'NA' else 'NA'
        cCN.Ccf_ci95_low_a2	        = float(list_sColumn[15]) if list_sColumn[15] != 'NA' else 'NA'
        cCN.Ccf_ci95_high_a2	    = float(list_sColumn[16]) if list_sColumn[16] != 'NA' else 'NA'
        cCN.LOH	                    = bool(list_sColumn[17])
        cCN.Homozygous_deletion	    = bool(list_sColumn[18])
        cCN.depMapID	            = list_sColumn[19]

        list_sOutput.append(cCN)
    #loop END: sReadLine

    #VS-Check
    if not list_sOutput: sys.exit('Empty List : basic_stat_ABSOLUTE_data : list_sOutput= %s' % len(list_sOutput))

    dict_sByTissue   = {}

    for cCN in list_sOutput:

        sCellLine = cCN.sSample.split('_')[0]
        sTissue   = '_'.join(cCN.sSample.split('_')[1:])

        if sTissue not in dict_sByTissue:
            dict_sByTissue[sTissue] = {}

        if sCellLine not in dict_sByTissue[sTissue]:
            dict_sByTissue[sTissue][sCellLine] = []

        dict_sByTissue[sTissue][sCellLine].append(cCN)
    #loop END: cCN

    for sTissue in dict_sByTissue:
        #print(sCancer, len(dict_sByTissue[sCancer]))

        for sCellLine in dict_sByTissue[sTissue]:

            if sCellLine not in list_sTargetCL: continue
            else:
                print('***********',  sCellLine, len(dict_sByTissue[sTissue][sCellLine]))

    pass
#def END: basic_stat_ABSOLUTE_data


def load_PE_input (sInFile):

    dict_sOutput = {}
    InFile       = open(sInFile, 'r')

    for sReadLine in InFile:
        ## File Format ##
        ## Target#  | Barcode | WT_Target | Edit_Target
        ## 181      | TTT.... | CTGCC..   | CTGCC...

        if sReadLine.startswith('Target'): continue ## SKIP HEADER

        list_sColumn = sReadLine.strip('\n').split('\t')

        cPE              = cPEData()
        cPE.nTargetNo    = int(list_sColumn[0])
        cPE.sBarcode     = list_sColumn[1]
        cPE.sWTSeq       = list_sColumn[2].upper()
        cPE.sAltSeq      = list_sColumn[3].upper()
        sKey             = cPE.sBarcode

        if sKey not in dict_sOutput:
            dict_sOutput[sKey] = ''
        dict_sOutput[sKey] = cPE
    #loop END:
    return dict_sOutput
#def END: load_PE_input


def load_NGS_files (sInFile):
    InFile       = open(sInFile, 'r')
    list_sOutput = [sReadLine.strip('\n') for sReadLine in InFile if not sReadLine.startswith('#')]
    InFile.close()

    dict_sOutput = {}
    for sFile in list_sOutput:

        sFileTag = '_'.join(sFile.split('_')[:3])

        if sFileTag not in dict_sOutput:
            dict_sOutput[sFileTag] = []
        dict_sOutput[sFileTag].append(sFile)
    #loop END: sFile

    return dict_sOutput
#def END: load_NGS_files


def split_fq_file (sWorkDir, sFastqTag, nCores):

    sOutDir    = '%s/split'                  % sWorkDir
    os.makedirs(sOutDir,exist_ok=True)

    sInFile    = '%s/%s.fastq'               % (sWorkDir, sFastqTag)
    print(sInFile)
    assert os.path.isfile(sInFile)

    sOutTag    = '%s/%s_fastq'               % (sOutDir, sFastqTag)

    sScript     = 'split --verbose '         # For Logging Purposes
    sScript    += '-l %s '                   % nLINE_CNT_LIMIT
    sScript    += '-a 3 '                    # Number of suffice places e.g. 001.fq

    sScript    += '--numeric-suffixes=1 '    # Start with number 1
    sScript    += '--additional-suffix=.fq ' # Add suffix .fq'
    sScript    += '%s %s_'                   % (sInFile, sOutTag)

    os.system(sScript)

    list_sFiles = os.listdir(sOutDir)
    sOutFile    = '%s/%s.filelist.txt'       % (sWorkDir, sFastqTag)
    OutFile     = open(sOutFile, 'w')
    for sFile in list_sFiles:
        sOut = '%s\n' % sFile
        OutFile.write(sOut)
    #loop END: sFile
    OutFile.close()

    return list_sFiles
#def END: split_fq_file


def get_line_cnt (sWorkDir, sFastqTag):

    sInFile = '%s/%s.filelist.txt' % (sWorkDir, sFastqTag)
    InFile = open(sInFile, 'r')
    list_sSplits = [sReadLine.strip('\n') for sReadLine in InFile if not sReadLine.startswith('#')]
    InFile.close()

    return list_sSplits
#def END: get_split_list

def get_split_list (sWorkDir, sFastqTag):

    sInFile = '%s/%s.filelist.txt' % (sWorkDir, sFastqTag)
    InFile = open(sInFile, 'r')
    list_sSplits = [sReadLine.strip('\n') for sReadLine in InFile if not sReadLine.startswith('#')]
    InFile.close()

    return list_sSplits
#def END: get_split_list


def mp_sort_by_barcode (nCores, sInDir, sFastqTag, sOutDir, sBarcodeFile, list_sSplits):

    list_sParameters = []
    for sSplitFile in list_sSplits:
        # HKK_191230_1.extendedFrags_|fastq_01|.fq
        sSplitTag = '_'.join(sSplitFile.split('.')[1].split('_')[-2:])
        sInFile   = '%s/split/%s' % (sInDir, sSplitFile)
        assert os.path.isfile(sInFile)

        sTempDir = '%s/%s' % (sOutDir, sSplitTag)
        os.makedirs(sTempDir, exist_ok=True)

        list_sParameters.append([sSplitTag, sInFile, sTempDir, sBarcodeFile])
    #loop END: sSplitFile

    p = mp.Pool(nCores)
    p.map_async(sort_by_barcode, list_sParameters).get()
    #p.map_async(determine_output, list_sParameters).get()

    #for sParameter in list_sParameters[:1]:
    #    determine_output(sParameter)
#def END: mp_sort_by_barcode


def sort_by_barcode (list_sParameters):

    print('Processing %s' % list_sParameters[1])

    sSplitTag      = list_sParameters[0]
    sInFastq       = list_sParameters[1]
    sTempOut       = list_sParameters[2]
    sBarcodeFile   = list_sParameters[3]
    sRE            = '[T]{7}'
    nBarcode3Cut   = 3
    nBarcode5Ext   = 18
    dict_sBarcodes = load_PE_input(sBarcodeFile)

    dict_sOutput   = {}
    InFile = open(sInFastq, 'r')
    for i, sReadLine in enumerate(InFile):

        if i % 4 == 0: sReadID = sReadLine.replace('\n', '')
        if i % 4 != 1: continue

        sNGSSeq = sReadLine.replace('\n', '').upper()

        for sReIndex in regex.finditer(sRE, sNGSSeq, overlapped=True):
            nIndexStart = sReIndex.start()
            nIndexEnd   = sReIndex.end()
            sBarcode    = sNGSSeq[nIndexStart+nBarcode3Cut:nIndexEnd+nBarcode5Ext]

            if nIndexStart > (len(sNGSSeq) / 2): continue # SKIP barcode in back of read

            ### Skip Non-barcodes ###
            try: cPE = dict_sBarcodes[sBarcode]
            except KeyError: continue
            #########################

            if sBarcode not in dict_sOutput:
                dict_sOutput[sBarcode] = []
            dict_sOutput[sBarcode].append([sReadID, sNGSSeq])

        #loop END: i, sReadLine
    #loop END: cPE
    InFile.close()
    ## Pickle Out ##
    sOutFile = '%s/%s.data' % (sTempOut, sSplitTag)
    OutFile = open(sOutFile, 'wb')
    pickle.dump(dict_sOutput, OutFile)
    OutFile.close()
    ###############

    '''
    for sKey in dict_sOutput:
        sOutFile = '%s/%s.txt' % (sTempOut, sKey)
        OutFile  = open(sOutFile, 'w')

        for sNGSSeq in dict_sOutput[sKey]:
            sOut = '%s\n' % sNGSSeq
            OutFile.write(sOut)
        #loop END: sNGSSeq
        OutFile.close()
    #loop END: sKey
    '''
#def END: check_barcode


def determine_output (list_sParameters):

    sSplitTag      = list_sParameters[0]
    sInFastq       = list_sParameters[1]
    sTempOut       = list_sParameters[2]
    sBarcodeFile   = list_sParameters[3]
    dict_cPE       = load_PE_input(sBarcodeFile)

    sInFile        = '%s/%s.data' % (sTempOut, sSplitTag)
    ## Pickle Load ##
    InFile         = open(sInFile, 'rb')
    dict_sBarcodes = pickle.load(InFile)
    InFile.close()
    print('%s dict_sBarcodes =%s' % (sSplitTag, len(dict_sBarcodes)))

    dict_sOutput   = {}
    for sBarcode in dict_sBarcodes:

        cPE      = dict_cPE[sBarcode]

        if sBarcode not in dict_sOutput:
            dict_sOutput[sBarcode] = {'WT': [], 'Alt': [], 'Other': []}

        nWTSize  = len(cPE.sWTSeq)
        nAltSize = len(cPE.sAltSeq)

        if sBarcode != 'TTTTGTACACACGCACGTATCG': continue
        print(sBarcode)

        for sReadID, sNGSSeq in dict_sBarcodes[sBarcode]:

            print(sReadID, sNGSSeq)

            nBarcodeS      = sNGSSeq.find(sBarcode)
            nBarcodeE      = nBarcodeS + len(sBarcode)
            sWTSeqCheck    = sNGSSeq[nBarcodeE:nBarcodeE+nWTSize]
            sAltSeqCheck   = sNGSSeq[nBarcodeE:nBarcodeE+nAltSize]

            if sWTSeqCheck == cPE.sWTSeq:
                dict_sOutput[cPE.sBarcode]['WT'].append(sReadID)

            elif sAltSeqCheck == cPE.sAltSeq:
                dict_sOutput[cPE.sBarcode]['Alt'].append(sReadID)

            elif sWTSeqCheck != cPE.sWTSeq and sAltSeqCheck != cPE.sAltSeq:
                dict_sOutput[cPE.sBarcode]['Other'].append(sReadID)
        #loop END: sReadID, sNGSSeq
    #loop END:
    list_sKeys = ['WT', 'Alt', 'Other']
    for sBarcode in dict_sOutput:

        if sBarcode != 'TTTTGTACACACGCACGTATCG': continue


        sOut = '%s\t%s\n'  % (sBarcode, '\t'.join([str(len(dict_sOutput[sBarcode][sType])) for sType in list_sKeys]))
        sOut2 = '%s\t%s\n' % (sBarcode, '\t'.join([','.join(dict_sOutput[sBarcode][sType]) for sType in list_sKeys]))

        print(sOut[:-1])
        print(sOut2[:-1])

    # loop END: sBarcode
#def END: determine_output_WTandEdited


def mp_sort_by_barcode_old (nCores, sFastqTag, sOutDir, sInFile, sOutFile, sOutFile2, list_cData):

    nBins            = 1
    nTotalCnt        = len(list_cData)
    list_nBin        = [[int(nTotalCnt * (i + 0) / nBins), int(nTotalCnt * (i + 1) / nBins)] for i in range(nBins)]
    list_sParameters = []

    sTempDir       = '%s/temp/%s' % (sOutDir, sFastqTag)
    os.makedirs(sTempDir, exist_ok=True)
    for nBinS, nBinE in list_nBin:
        sTempOut        = '%s/tempout.%s-%s.txt'   % (sTempDir, nBinS, nBinE)
        sReadIDOut      = '%s/readIDout.%s-%s.txt' % (sTempDir, nBinS, nBinE)
        list_cData_bin  = list_cData[nBinS:nBinE]
        list_sParameters.append([list_cData_bin, sInFile, sTempOut, sReadIDOut])
    #loop END: nBinS, nBinE

    #for sParameters in list_sParameters[:1]:
    #    check_barcode(sParameters)

    #p = mp.Pool(nCores)
    #p.map_async(check_barcode, list_sParameters).get()

    ## Combine Output Files ##
    #sCmd        = 'cat %s/tempout.*.txt   > %s;'    % (sTempDir, sOutFile)
    #sCmd2       = 'cat %s/readIDout.*.txt > %s;'    % (sTempDir, sOutFile2)
    #os.system(sCmd)
    #os.system(sCmd2)
#def END: mp_sort_by_barcode


def check_barcode_old (sParameter):

    list_cPE       = sParameter[0]
    sInFastq       = sParameter[1]
    sTempOut       = sParameter[2]
    sReadIDOut     = sParameter[3]
    dict_sOutput   = {}
    InFile = open(sInFastq, 'r')
    for i, sReadLine in enumerate(InFile):

        if i % 4 == 0: sReadID = sReadLine.replace('\n', '')
        if i % 4 != 1: continue

        for n, cPE in enumerate(list_cPE):

            nWTSize  = len(cPE.sWTSeq)
            nAltSize = len(cPE.sAltSeq)

            if cPE.sBarcode not in dict_sOutput:
                dict_sOutput[cPE.sBarcode] = {'WT':[], 'Alt':[],'Other':[]}

            sNGSSeq       = sReadLine.replace('\n','').upper()
            nBarcodeS     = sNGSSeq.find(cPE.sBarcode)
            nBarcodeE     = nBarcodeS + len(cPE.sBarcode)
            sBarcodeCheck = sNGSSeq[nBarcodeS:nBarcodeE]

            if nBarcodeS < 0: continue # Barcode Not Found
            assert cPE.sBarcode == sBarcodeCheck

            sWTSeqCheck    = sNGSSeq[nBarcodeE:nBarcodeE+nWTSize]
            sAltSeqCheck   = sNGSSeq[nBarcodeE:nBarcodeE+nAltSize]

            if sWTSeqCheck == cPE.sWTSeq:
                dict_sOutput[cPE.sBarcode]['WT'].append(sReadID)

            elif sAltSeqCheck == cPE.sAltSeq:
                dict_sOutput[cPE.sBarcode]['Alt'].append(sReadID)

            elif sWTSeqCheck != cPE.sWTSeq and sAltSeqCheck != cPE.sAltSeq:
                dict_sOutput[cPE.sBarcode]['Other'].append(sReadID)
        #loop END: i, sReadLine
    #loop END: cPE
    InFile.close()

    OutFile     = open(sTempOut, 'w')
    OutFile2    = open(sReadIDOut, 'w')
    list_sKeys  = ['WT', 'Alt', 'Other']
    for sBarcode in dict_sOutput:
        sOut   = '%s\t%s\n' % (sBarcode, '\t'.join([str(len(dict_sOutput[sBarcode][sType])) for sType in list_sKeys]))
        sOut2  = '%s\t%s\n' % (sBarcode, '\t'.join([','.join(dict_sOutput[sBarcode][sType]) for sType in list_sKeys]))
        OutFile.write(sOut)
        OutFile2.write(sOut2)
    #loop END: sBarcode
    OutFile.close()
    OutFile2.close()
#def END: check_barcode


def analyze_PE_KO_output (sInFile, sInFile2):

    InFile       = open(sInFile, 'r')
    list_sOutput = []
    for sReadLine in InFile:
        list_sColumn = sReadLine.strip('\n').split('\t')
        sBarcode     = list_sColumn[0]
        nWT          = int(list_sColumn[1])
        nEdited      = int(list_sColumn[2])
        nOther       = int(list_sColumn[3])
        list_sOutput.append([sBarcode, nWT, nEdited, nOther])
    #loop END: sReadLine
    InFile.close()

    InFile = open(sInFile2, 'r')
    dict_sOutput = {}
    for sReadLine in InFile:
        list_sColumn = sReadLine.strip('\n').split('\t')
        sBarcode     = list_sColumn[0]
        sWT          = list_sColumn[1]
        sAlt         = list_sColumn[2]
        sOther       = list_sColumn[3]

        if sBarcode not in dict_sOutput:
            dict_sOutput[sBarcode] = []
        dict_sOutput[sBarcode] = {'WT':sWT, 'Alt':sAlt, 'Other':sOther}
    #loop END: sReadLine

    return list_sOutput, dict_sOutput
#def END: analyze_PE_KO_output


def load_HGNC_reference (sInFile):
    dict_sOutput = {}
    InFile       = open(sInFile, 'r')

    for sReadLine in InFile:
        ## File Format ##
        #HGNC:ID	Approved symbol
        #HGNC:5	A1BG

        if sReadLine.startswith('HGNC:ID'): continue

        list_sColumn = sReadLine.strip('\n').split('\t')

        sHGNCID      = list_sColumn[0].split(':')[1]
        sGeneSym     = list_sColumn[1].upper()

        if sHGNCID not in dict_sOutput:
            dict_sOutput[sHGNCID] = ''
        dict_sOutput[sHGNCID] = sGeneSym
    #loop END: sReadLine
    return dict_sOutput
#def END: load_HGNC_reference


def load_target_genes (sInFile):
    dict_sOutput = {}
    InFile       = open(sInFile, 'r')

    for sReadLine in InFile:
        ## File Format ##
        ## Group        GeneName    HGNCID
        ## 6-TG_related HPRT1	5157

        if sReadLine.startswith('Group'): continue
        if sReadLine.startswith('#'): continue

        list_sColumn = sReadLine.strip('\n').split('\t')

        sGroup       = list_sColumn[0]
        sGeneSym     = list_sColumn[1].upper()
        sHGNCID      = list_sColumn[2]

        if sGeneSym not in dict_sOutput:
            dict_sOutput[sGeneSym] = ''
        dict_sOutput[sGeneSym] = [sHGNCID, sGroup]
    #loop END: sReadLine

    return dict_sOutput
#def END: load_target_genes


def load_target_genes_posctrl (sInFile):
    dict_sOutput = {}
    InFile       = open(sInFile, 'r')

    for sReadLine in InFile:
        ## File Format ##
        ## HGNCSymbol	HGNCID	    EntrezID	ENSEMBLEID
        ## 6ERH	        HGNC:3447	2079	    ENSG00000100632

        if sReadLine.startswith('HGNCSymbol'): continue

        list_sColumn = sReadLine.strip('\n').split('\t')

        sGeneSym     = list_sColumn[0]
        sHGNCID      = list_sColumn[1].upper()
        sEntrezID    = list_sColumn[2]
        sEnsembleID  = list_sColumn[3]

        if sGeneSym not in dict_sOutput:
            dict_sOutput[sGeneSym] = ''
        dict_sOutput[sGeneSym] = [sHGNCID, sEntrezID, sEnsembleID]
    #loop END: sReadLine

    return dict_sOutput
#def END: load_target_genes_posctrl


def load_clinvar_PE (sInFile, nBufferSize, cGenome):

    list_sOutput  = []
    InFile        = open(sInFile, 'r')
    for sReadLine in InFile:

        if sReadLine.startswith('#'): continue  # SKIP Information Headers
        list_sColumn = sReadLine.strip('\n').split('\t')

        cVCF         = cVCFData()
        cVCF.sChrID  = 'chr%s' % list_sColumn[0]

        if list_sColumn[0].startswith('MT'): continue
        if list_sColumn[0].startswith('NW'): continue

        try: cVCF.nPos = int(list_sColumn[1])
        except ValueError: continue

        cVCF.nStartPos  = cVCF.nPos - nBufferSize
        cVCF.nEndPos    = cVCF.nPos + nBufferSize
        cVCF.sDBSNP_ID  = list_sColumn[2]
        cVCF.sRefNuc    = list_sColumn[3]
        cVCF.sAltNuc    = list_sColumn[4] if list_sColumn[4] != '.' else ''
        cVCF.nAltLen    = int(list_sColumn[5])

        cVCF.sAltType   = list_sColumn[6]
        cVCF.sGeneSym   = list_sColumn[7]
        cVCF.sRefSeq    = cGenome.fetch(cVCF.sChrID, cVCF.nStartPos - 1, cVCF.nEndPos).upper()

        list_sOutput.append(cVCF)
    #loop END: sLine
    return list_sOutput
#def END: load_clinvar_PE


def get_guides_from_genome (sOutputDir, dict_sHGNC, dict_cRefGene, cGenome, dict_sTargetGenes):

    ## Flanking Sizes ##
    nUpFlank         = 24
    nDownFlank       = 23
    nBufferSize      = 100
    dict_sRE         = {'+': '[ACGT]GG',
                        '-': 'CC[ACGT]'}
    list_cFinal_Ref  = []
    list_sNotFound   = []
    for sGeneSym in dict_sTargetGenes:

        sHGNCID, sEntrezID, sEnsembleID = dict_sTargetGenes[sGeneSym]


        try: list_cRef = dict_cRefGene[sGeneSym] # Multiple Transcripts per GeneSym
        except KeyError:
            try:
                sGeneSym_hgnc = dict_sHGNC[sHGNCID]
                list_cRef     = dict_cRefGene[sGeneSym_hgnc]

            except KeyError:
                list_sNotFound.append([sGeneSym, sHGNCID, sEntrezID, sEnsembleID])
                continue
            #try END:
        #try END:

        if len(list_cRef) == 1:
            cRef = list_cRef[0]
        else:
            list_cRef_sorted = sorted(list_cRef, key=lambda c:int(c.sNMID.replace('NM_', '')))
            cRef = list_cRef_sorted[0]
        #if END:

        #print(sGeneSym, cRef.sStrand, sHGNCID, sGroupName, cRef.nTxnStartPos, cRef.nTxnEndPos, cRef.nORFStartPos, cRef.nORFEndPos)

        cRef.sHGNCID       = sHGNCID
        cRef.nTxnSize      = cRef.nTxnEndPos - cRef.nTxnStartPos
        cRef.dict_nCDSLen  = {}
        cRef.dict_sExonSeq = {}
        cRef.dict_nExonPos = {}
        cRef.dict_sGuides  = {}
        cRef.nCDSSize      = sum([nORF_E-nORF_S for nORF_S, nORF_E in zip(cRef.nORFStartList, cRef.nORFEndList)])
        assign_exon_features (cRef, cGenome)

        list_sExonKey      = sorted(list(cRef.dict_nCDSLen.keys()), key=lambda e:int(e.split('-')[1]))
        check_PAM_v2(cRef, cGenome, dict_sRE, list_sExonKey, nUpFlank, nDownFlank, nBufferSize)

        list_cFinal_Ref.append(cRef)
    #loop END: sGeneSym

    print('Target Genes', len(dict_sTargetGenes))
    print('list_cFinal_Ref', len(list_cFinal_Ref))

    sOutFile = '%s/20201015_PEKO_PosCtrl.txt'          % sOutputDir
    output_full_data (sOutFile, list_cFinal_Ref)

    sOutFile = '%s/20201015_PEKO_PosCtrl_Notfound.txt' % sOutputDir
    OutFile  = open(sOutFile, 'w')
    for sGeneSym, sHGNCID, sEntrezID, sEnsembleID in list_sNotFound:
        sOut = '%s,%s,%s,%s\n' % (sGeneSym, sHGNCID, sEntrezID, sEnsembleID)
        OutFile.write(sOut)
    #loop END: sGeneSym, sHGNCID, sGroupName
    OutFile.close()
#def END: get_guides


def assign_exon_features (cRef, cGenome):

    if cRef.sStrand == '+':
        list_nORFS = cRef.nORFStartList
        list_nORFE = cRef.nORFEndList
    else:
        list_nORFS = reversed(cRef.nORFStartList)
        list_nORFE = reversed(cRef.nORFEndList)

    nCDSLen = 0
    for nORF_S, nORF_E in zip(list_nORFS, list_nORFE):

        for i, nExonE in enumerate(cRef.list_nExonE):
            if nORF_E <= nExonE:
                nExonCnt = (i+1) if cRef.sStrand == '+' else (cRef.nExonCount - i)
                break
        #loop END: i, nExonE

        sKey     = 'Exon-%s' % (nExonCnt)
        nCDSLen += nORF_E - nORF_S

        if sKey not in cRef.dict_nCDSLen:
            cRef.dict_nCDSLen[sKey] = ''

        if sKey not in cRef.dict_sExonSeq:
            cRef.dict_sExonSeq[sKey] = ''

        if sKey not in cRef.dict_nExonPos:
            cRef.dict_nExonPos[sKey] = ''

        cRef.dict_nCDSLen[sKey]  = nCDSLen
        cRef.dict_sExonSeq[sKey] = cGenome.fetch(cRef.sChrID, nORF_S - 1, nORF_E, cRef.sStrand).upper()
        cRef.dict_nExonPos[sKey] = [nORF_S, nORF_E]
    #loop END: nORF_S, nORF_E
#def END: assign_exon_features


def check_PAM_v2(cRef, cGenome, dict_sRE, list_sExonKey, nUpFlank, nDownFlank, nBufferSize):

    for sStrand in ['+', '-']:

        nGuideCnt  = 0
        sRE        = dict_sRE[sStrand]
        nMinExonNo = min([int(sExonKey.split('-')[1]) for sExonKey in list_sExonKey])

        for sExonKey in list_sExonKey:

            nExonCnt = int(sExonKey.split('-')[1])

            if sExonKey not in cRef.dict_sGuides:
                cRef.dict_sGuides[sExonKey] = {}

            sFullSeq       = cRef.dict_sExonSeq[sExonKey] if cRef.sStrand == '+' \
                else reverse_complement(cRef.dict_sExonSeq[sExonKey])

            nExonS, nExonE = cRef.dict_nExonPos[sExonKey]
            nCDSLen        = cRef.dict_nCDSLen[sExonKey]

            for sReIndex in regex.finditer(sRE, sFullSeq, overlapped=True):

                nGuideCnt   += 1
                nIndexStart  = sReIndex.start()
                nIndexEnd    = sReIndex.end()
                sPAM         = sFullSeq[nIndexStart:nIndexEnd]

                if sStrand == '+':
                    nSeqStart    = nIndexStart  - nUpFlank
                    nSeqEnd      = nIndexEnd    + nDownFlank
                    nGenomicS    = nExonS       + nSeqStart
                    nGenomicE    = nExonS       + nSeqEnd - 1
                    nCutSitePos  = nGenomicE    - 9

                    if nGenomicS < nExonS or nGenomicE > nExonE:
                        sTarSeq = cGenome.fetch(cRef.sChrID, nGenomicS - 1, nGenomicE, sStrand).upper()
                    else: sTarSeq = sFullSeq[nSeqStart:nSeqEnd]

                    nBufferS     = nGenomicS - nBufferSize
                    nBufferE     = nGenomicE + nBufferSize
                    sBufferSeq   = cGenome.fetch(cRef.sChrID, nBufferS - 1, nBufferE, sStrand).upper()

                else:
                    nSeqStart    = nIndexStart - nDownFlank
                    nSeqEnd      = nIndexEnd   + nUpFlank
                    nGenomicS    = nExonS      + nSeqStart
                    nGenomicE    = nExonS      + nSeqEnd - 1
                    nCutSitePos  = nGenomicS   + 9 - 1

                    if nGenomicS < nExonS or nGenomicE > nExonE:
                        sTarSeq = cGenome.fetch(cRef.sChrID, nGenomicS - 1, nGenomicE, sStrand).upper()
                    else: sTarSeq = reverse_complement(sFullSeq[nSeqStart:nSeqEnd])

                    nBufferS     = nGenomicS - nBufferSize
                    nBufferE     = nGenomicE + nBufferSize
                    sBufferSeq   = cGenome.fetch(cRef.sChrID, nBufferS - 1, nBufferE, sStrand).upper()
                    sPAM         = reverse_complement(sPAM)
                #if END:

                nCDSLen      = 0 if nExonCnt == nMinExonNo else cRef.dict_nCDSLen['Exon-%s' % (nExonCnt - 1)]
                nCutIndex    = nCDSLen + nCutSitePos - nExonS
                nCutPercent  = '%0.2f%%' % ((nCutIndex  / cRef.nCDSSize) * 100)

                #print(sExonKey, nGenomicS, nGenomicE, nExonE- nExonS, nCDSLen, nCutIndex, nCutSitePos, nCutPercent, sTarSeq, sPAM, sStrand)
                sKey = 'pegRNA-%s.%s-%s.%s' % (nGuideCnt, nGenomicS, nGenomicE, sStrand)

                ## All PBS and RT Seq for HKK 20201015 VUS Positive Controls ##
                sPBSSeq    = sTarSeq[nUpFlank - 13:nUpFlank - 3]
                sRTSeq     = sTarSeq[nUpFlank - 3:nUpFlank + 1] + 'CC' + sTarSeq[nUpFlank + 1:nUpFlank + 9]

                print(len(sPBSSeq), len(sRTSeq))

                if sKey not in cRef.dict_sGuides[sExonKey]:
                    cRef.dict_sGuides[sExonKey][sKey] = ''
                cRef.dict_sGuides[sExonKey][sKey] = [sTarSeq, sPBSSeq, sRTSeq]
            #loop END: sReIndex
        #loop END: sExonKey
    #loop END: sStrand
#def END: check_PAM


def output_full_data (sOutFile, list_cRef):

    OutFile = open(sOutFile, 'w')

    for i, cRef in enumerate(list_cRef):

        list_sExonKeys = list(cRef.dict_sGuides.keys())

        for sExonKey in list_sExonKeys:

            for sGuideKey in cRef.dict_sGuides[sExonKey]:
                sGuideSeq, sPBSSeq, sRTSeq = cRef.dict_sGuides[sExonKey][sGuideKey]

                sOut = '%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
                       % (cRef.sGeneSym, cRef.sNMID, cRef.sChrID, cRef.sStrand,
                          sExonKey, sGuideKey, sGuideSeq, sPBSSeq, sRTSeq)
                OutFile.write(sOut)
            #loop END: sGuideKey
        #loop END: sExonKey
    #loop END: i, cRef
    OutFile.close()
#def END: output_full_data


def get_guides_from_GUIDES (sInputDir, sOutputDir, dict_sHGNC, dict_cRefGene, cGenome, dict_sTargetGenes):

    ## Find Target Gene Data ##  *Run only once
    #extract_targetgene_GUIDESdata (sInputDir, dict_sTargetGenes)

    ## LiftOver Cmd ## *hg19 to hg38
    #run_liftover (sInputDir)

    ## Extract updated guides ##
    nFlankSize   = 100
    sGuideFile   = '%s/cGuidesData.data' % sInputDir
    InFile       = open(sGuideFile, 'rb')
    list_cGuide  = pickle.load(InFile)
    InFile.close()
    print('list_cGuide', len(list_cGuide))

    sLiftOverKey = '%s/LiftOver/TargetGUIDES_PAMPos.liftover.bed' % sInputDir
    dict_sKey    = {sLine.strip('\n').split('\t')[3]:sLine.strip('\n').split('\t')[1] for sLine in open(sLiftOverKey)}

    sOutFile     = '%s/TargetGene_GUIDES.txt' % sOutputDir
    OutFile      = open(sOutFile, 'w')

    for cGuide in list_cGuide:
        try: sPAMPos_hg38 = int(dict_sKey[cGuide.sGuideID])
        except KeyError: continue

        nStart        = sPAMPos_hg38 - nFlankSize + 1
        nEnd          = sPAMPos_hg38 + nFlankSize
        print(cGuide.sChrom, nStart, nEnd)

        try:
            sGuideSeq     = cGenome.fetch('chr%s' % cGuide.sChrom, nStart, nEnd, '+')
            sGuideSeq_rev = cGenome.fetch('chr%s' % cGuide.sChrom, nStart, nEnd, '-')
        except AssertionError: continue

        sOut = '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' % \
               (cGuide.sGuideID, cGuide.sGeneSym, cGuide.sEnsemblID, cGuide.sChrom, cGuide.sGuideSeq,
                cGuide.sGuideSeq_ext, cGuide.sPAMPos, cGuide.fOnTarget, cGuide.nExonNo, cGuide.sTargetExon,
                cGuide.sOfftargetMat, cGuide.sOffScore, cGuide.sDomain, sGuideSeq, sPAMPos_hg38)
        OutFile.write(sOut)
    #loop END: cGuide
#def END: get_guides


def extract_targetgene_GUIDESdata (sInputDir, dict_sTargetGenes):

    sGUIDES_Dir       = '%s/GUIDES_Data' % sInputDir
    list_sGuideFiles  = (os.listdir(sGUIDES_Dir))
    list_sTargetGenes = list(dict_sTargetGenes.keys())

    list_sGuideInfo   = []
    for sGuideFile in list_sGuideFiles:

        sInFile      = '%s/%s' % (sGUIDES_Dir, sGuideFile)

        print('Loading %s' % sInFile)
        list_sGuideInfo += load_GUIDES_data (sInFile, list_sTargetGenes)

    #loop END: sGuideFile

    print('GUIDES Data', len(list_sGuideInfo))

    ## Output format for LiftOver ##
    sOutFile = '%s/LiftOver/TargetGUIDES_PAMPos.bed' % sInputDir
    OutFile  = open(sOutFile, 'w')

    for cGuides in list_sGuideInfo:
        sOut = 'chr%s %s %s %s\n' % (cGuides.sChrom, cGuides.sPAMPos, cGuides.sPAMPos+1, cGuides.sGuideID)
        OutFile.write(sOut)
    #loop END: cGuides
    OutFile.close()

    ## Pickle Out ##
    sOutFile = '%s/cGuidesData.data' % sInputDir
    OutFile = open(sOutFile, 'wb')
    pickle.dump(list_sGuideInfo, OutFile)
    OutFile.close()
#def END: extract_targetgene_GUIDESdata


def load_GUIDES_data (sInFile, list_sTargetGenes):

    InFile       = open(sInFile, 'r')
    list_sOutput = []

    for sReadLine in InFile:
        #GUIDE_ID	                GUIDES_sg000001
        #Gene	                    A4GNT
        #Ensembl_ID	                ENSG00000118017.3
        #Sequence	                AGGATATGTTCAGACACCTG
        #GuideSeq_10bp	            GGGTGTAAGAAGGATATGTTCAGACACCTGAGGAGGTCGCTCA
        #Chromosome	                3
        #PAM_position_in_chromosome	137843388     *0-based
        #On-target_efficiency	    0.770294237
        #Exon	                    3
        #Targets_last_exon	        TRUE
        #10bp_off-target_match	    FALSE
        #Off-target_score	        0
        #Protein_domain	            Gb3_synth

        if sReadLine.startswith('GUIDE ID'): continue

        list_sColumn          = sReadLine.strip('\n').split(',')

        cGuides               = cGUIDESData()
        cGuides.sGuideID      = list_sColumn[0]
        cGuides.sGeneSym      = list_sColumn[1]
        cGuides.sEnsemblID    = list_sColumn[2]
        cGuides.sGuideSeq     = list_sColumn[3]
        cGuides.sGuideSeq_ext = list_sColumn[4]
        cGuides.sChrom        = list_sColumn[5]
        cGuides.sPAMPos       = int(list_sColumn[6])
        cGuides.fOnTarget     = float(list_sColumn[7])
        cGuides.nExonNo       = int(list_sColumn[8])
        cGuides.sTargetExon   = list_sColumn[9]
        cGuides.sOfftargetMat = list_sColumn[10]
        cGuides.sOffScore     = list_sColumn[11]
        cGuides.sDomain       = list_sColumn[12]

        if cGuides.sGeneSym not in set(list_sTargetGenes): continue

        list_sOutput.append(cGuides)
    #loop END: sReadLine
    InFile.close()

    return list_sOutput
#def END: load_GUIDES_data


def run_liftover (sInputDir):

    sLiftDir    = '%s/LiftOver'                          % sInputDir
    sInBed      = '%s/TargetGUIDES_PAMPos.bed'           % sLiftDir
    sChainFile  = '%s/hg19/hg19ToHg38.over.chain.gz'     % sREF_DIR
    sOutBed     = '%s/TargetGUIDES_PAMPos.liftover.bed'  % sLiftDir
    sOutTemp    = '%s/TargetGUIDES_PAMPos.liftover.txt'  % sLiftDir

    sScript     = '%s %s %s %s %s' % (sLIFTOVER, sInBed, sChainFile, sOutBed, sOutTemp)
    os.system(sScript)
#def END: run_liftover


def run_liftover_VUS (sLiftDir, sFileTag):

    sInBed      = '%s/%s_hg19.bed'                      % (sLiftDir, sFileTag)
    sChainFile  = '%s/hg19/hg19ToHg38.over.chain.gz'    % sREF_DIR
    sOutBed     = '%s/%s_hg38.bed'                      % (sLiftDir, sFileTag)
    sOutTemp    = '%s/%s_hg19.liftover.txt'             % (sLiftDir, sFileTag)

    sScript     = '%s %s %s %s %s' % (sLIFTOVER, sInBed, sChainFile, sOutBed, sOutTemp)
    os.system(sScript)
#def END: run_liftover


def rank_candidates_by_gene (sInFile):

    InFile       = open(sInFile, 'r')
    dict_sOutput = {}
    nTop         = 20
    fMinScore    = 20

    for sReadLine in InFile:

        list_sColumn    = sReadLine.strip('\n').split('\t')

        sGeneSym        = list_sColumn[0]
        fAvgScore       = float(list_sColumn[1])
        spegRNASeq_pri  = list_sColumn[2]
        fScore_pri      = float(list_sColumn[3])
        spegRNASeq_2nd  = list_sColumn[4]
        fScore_2nd      = float(list_sColumn[5])
        sRT_PBSSeq      = list_sColumn[6]

        if fScore_pri < fMinScore: continue
        if fScore_2nd < fMinScore: continue

        if sGeneSym not in dict_sOutput:
            dict_sOutput[sGeneSym] = []
        dict_sOutput[sGeneSym].append([fAvgScore, spegRNASeq_pri, fScore_pri, spegRNASeq_2nd, fScore_2nd, sRT_PBSSeq])
    #loop END: sReadLine
    InFile.close()

    print('Gene Count', len(dict_sOutput))
    list_sOutput = []
    for sGeneSym in dict_sOutput:

        list_sData = dict_sOutput[sGeneSym]
        list_sData = sorted(list_sData, key=lambda e:e[0], reverse=True)

        for i, sData in enumerate(list_sData[:nTop]):

            sGeneName = '%s_%s' % (sGeneSym, (i+1))
            sOut      = [sGeneName] + sData
            list_sOutput.append(sOut)
        #loop END: i, sData
    #loop END: sGeneSym
    print('Gene Check', len(list_sOutput))

    return list_sOutput
#def END: rank_candidates_by_gene


def VUS_basicstat_makeinput (sInputDir, sOutputDir):

    dict_sCLNVCKey = {'Insertion':  'insertion',
                      'Duplication':'insertion',

                      'Deletion':      'deletion',

                      'Inversion':'substitution',
                      'single_nucleotide_variant':'substitution'}

    ## RefSeq Database ##
    sRefFlat      = '%s/%s_refFlat_step4.txt' % (sGENOME_DIR, sGENOME)
    dict_cRefGene = parse_refflat_line(sRefFlat, 'dict', 'Gene')

    sClinVarFile  = '%s/ClinVar/20200420_Clinvar.vcf' % sInputDir
    list_cVCF     = cVCF_parse_vcf_files_clinvar(sClinVarFile)

    print('Clinvar Count', len(list_cVCF))
    print('Refseq Data Count', len(dict_cRefGene))

    dict_sCLINSIG  = {}
    dict_sGeneInfo = {}
    nNoCLINSIG     = 0
    nNoGeneInfo    = 0
    nNoRefGene     = 0

    list_sFinalVCF = []
    for cVCF in list_cVCF:

        ## Filter 1: No CLINSIG Info ##
        try: sCLNSIG = cVCF.dict_sInfo['CLNSIG']
        except KeyError:
            nNoCLINSIG += 1
            continue

        if sCLNSIG not in dict_sCLINSIG:
            dict_sCLINSIG[sCLNSIG] = 0
        dict_sCLINSIG[sCLNSIG] += 0

        ## Filter 2: Target CLNSIG ##
        if sCLNSIG != 'Uncertain_significance': continue

        ## Filter 3: No Gene Info ##
        try: sGeneInfo = cVCF.dict_sInfo['GENEINFO']
        except KeyError:
            nNoGeneInfo += 1
            continue

        cVCF.sGeneSym   = [sGeneSymID.split(':')[0] for sGeneSymID in sGeneInfo.split('|')][0].upper()

        try: cRef   = dict_cRefGene[cVCF.sGeneSym]
        except KeyError:
            nNoRefGene += 1

            if cVCF.sGeneSym not in dict_sGeneInfo:
                dict_sGeneInfo[cVCF.sGeneSym] = 0
            dict_sGeneInfo[cVCF.sGeneSym] += 1
            continue

        ## Filter 4: Assign proper alt type and length
        if max([len(cVCF.sRefNuc), len(cVCF.sAltNuc)]) > 4: continue
        if cVCF.dict_sInfo['CLNVC'] == 'Variation': continue

        if not determine_alttype_altlen (cVCF, dict_sCLNVCKey): continue

        if cVCF.sAltType == 'substitution':
            if len(cVCF.sAltNuc) > 3: continue
        else:
            if len(cVCF.sAltNuc) > 4: continue

        list_sFinalVCF.append(cVCF)
    #loop END: cVCF

    print('dict_sGeneInfo', len(dict_sGeneInfo))
    print('nNoCLINSIG', nNoCLINSIG)
    print('nNoGeneInfo', nNoGeneInfo)
    print('nNoRefGene', nNoRefGene)


    sOutFile = '%s/ClinVar/20200909_ClinVar_hg38_NoGeneSymMatch.txt' % sInputDir
    OutFile  = open(sOutFile, 'w')
    for sGeneSym in dict_sGeneInfo:
        sOut = '%s\n' % sGeneSym
        OutFile.write(sOut)
    #loop END: sGeneSym
    OutFile.close()

    ## Make input for pegRNA design VUS##
    sOutFile = '%s/ClinVar/20200909_ClinVar_hg38.txt' % sInputDir
    OutFile  = open(sOutFile, 'w')

    sHeader  = 'CHROM\tPOS\tID\tREF\tALT\tmut_length\tCLNVC\tGENEINFO\n'
    OutFile.write(sHeader)

    for cVCF in list_sFinalVCF:

        nChrNo   = cVCF.sChrID.replace('chr', '')

        sOut   = '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                 % (nChrNo, cVCF.nPos, cVCF.sDBSNP_ID,  cVCF.sRefNuc, cVCF.sAltNuc,
                    cVCF.nAltLen, cVCF.sAltType, cVCF.sGeneSym)
        OutFile.write(sOut)
    #loop END: cVCF
    OutFile.close()
#def END: VUS_basicstat_makeinput


def check_refseq_match (cVCF, sGeneInfo, dict_cRefGene):

    sGeneSym   = [sGeneSymID.split(':')[0] for sGeneSymID in sGeneInfo.split('|')][0]

    try:
        cRef   = dict_cRefGene[sGeneSym]

    except KeyError:
        return  None
#def END: check_refseq_match


def determine_alttype_altlen (cVCF, dict_sCLNVCKey):

    if cVCF.dict_sInfo['CLNVC'] in ['Microsatellite', 'Indel']:
        if len(cVCF.sRefNuc) > len(cVCF.sAltNuc):
            cVCF.sAltType = 'deletion'
            cVCF.nAltLen  = len(cVCF.sRefNuc) - 1

            if cVCF.sRefNuc[0] != cVCF.sAltNuc[0]:
                return None # ex) GAGA -> TCT


        elif len(cVCF.sRefNuc) < len(cVCF.sAltNuc):
            cVCF.sAltType = 'insertion'
            cVCF.nAltLen  = len(cVCF.sAltNuc) - 1

            if cVCF.sRefNuc[0] != cVCF.sAltNuc[0]:
                return None # ex) GAGA -> TCT

        else:
            cVCF.sAltType = 'substitution'
            cVCF.nAltLen  = len(cVCF.sAltNuc)
    else:
        cVCF.sAltType = dict_sCLNVCKey[cVCF.dict_sInfo['CLNVC']]
        if dict_sCLNVCKey[cVCF.dict_sInfo['CLNVC']] == 'deletion':
            cVCF.nAltLen = len(cVCF.sRefNuc) - 1
        elif dict_sCLNVCKey[cVCF.dict_sInfo['CLNVC']] == 'insertion':
            cVCF.nAltLen = len(cVCF.sAltNuc) - 1
        else:
            cVCF.nAltLen = len(cVCF.sAltNuc)

    return 1
#def END: determine_alttype_altlen


def load_PAM_indel_freq (sInFile):

    dict_sOutput = {}
    InFile       = open(sInFile, 'r')

    for sReadLine in InFile:

        list_sColumn = sReadLine.strip('\n').split('\t')

        sPAM         = list_sColumn[0]
        sAA          = list_sColumn[1]
        fIndelFreq   = float(list_sColumn[2])
        sFraction    = list_sColumn[3]

        if sPAM not in dict_sOutput:
            dict_sOutput[sPAM] = ''
        dict_sOutput[sPAM] = fIndelFreq

    #loop END: sReadLine

    return dict_sOutput
#def END: load_PAM_indel_freq


def VUS_library_input_get_genomic_position(sInputDir):

    ## Genome FASTA ##
    sGenomeFile       = '%s/%s/%s.fa' % (sREF_DIR, sGENOME, sGENOME)
    cGenome           = cFasta(sGenomeFile)

    ## RefSeq Database ##
    sRefFlat          = '%s/%s_refFlat_step4.txt' % (sGENOME_DIR, sGENOME)
    list_cRefGene     = parse_refflat_line(sRefFlat, 'list', 'Gene')
    dict_cRefGene     = filter_refflat(list_cRefGene)
    nTargetLength     = 20

    sFileTag      = '20201026_VUS_Input'
    sInFile       = '%s/%s_edited.txt' % (sInputDir, sFileTag)
    InFile        = open(sInFile, 'r')
    list_sOutput  = []
    for sReadLine in InFile:

        if sReadLine.startswith('GeneID'): continue

        list_sColumn = sReadLine.strip('\n').split('\t')

        sGeneID      = list_sColumn[0]
        sGeneSym     = list_sColumn[0].split('_')[0].upper()
        sAAChange    = list_sColumn[0].split('_')[1]
        sPath        = list_sColumn[1]
        sWTSeq       = list_sColumn[2]
        sEditedSeq   = list_sColumn[3]
        nAltIndex    = [i for i, (sWT, sEdit) in enumerate(zip(sWTSeq, sEditedSeq)) if sWT != sEdit][0]
        cRef         = dict_cRefGene[sGeneSym][0]

        sWTNuc       = sWTSeq[nAltIndex].upper() if cRef.sStrand == '+' else reverse_complement(sWTSeq[nAltIndex].upper())
        sAltNuc      = sEditedSeq[nAltIndex].upper() if cRef.sStrand == '+' else reverse_complement(sEditedSeq[nAltIndex].upper())

        #if sGeneSym != 'ARHGEF1': continue

        nAltPos      = obtain_genomic_pos (cRef, cGenome, sWTSeq, nTargetLength, nAltIndex)

        list_sOutput.append([cRef.sChrID.replace('chr', ''), nAltPos, sAAChange, sWTNuc, sAltNuc, len(sAltNuc), 'substitution', sGeneSym])
    #loop END: sReadLine
    InFile.close()

    ## Output in VCF format for VUS ##
    output_VCF_format (sInputDir, sFileTag, list_sOutput)
#def END: VUS_library_input_get_genomic_position


def output_VCF_format (sInputDir, sFileTag, list_sOutput):

    sOutFile = '%s/ClinVar/%s_%s.txt' % (sInputDir, sFileTag, sGENOME)
    OutFile  = open(sOutFile, 'w')

    for sOutput in list_sOutput:
        sOut = '\t'.join([str(sData) for sData in sOutput])
        OutFile.write('%s\n' % sOut)
    #loop END: sData
    OutFile.close()
#def END: output_VCF_format


def determine_clinvar_mutations_VUS (sInputDir, sOutputDir, bTestRun):

    ## RT template length ~20
    ## PAM (+5, 6)에 silent mutation 넣기
    ## PBS length = 13으로 고정

    sGenome         = 'hg38'
    sGenomeFile     = '%s/%s.fa' % (sGENOME_DIR, sGenome)
    cGenome         = cFasta(sGenomeFile)

    #sFileTag        = '20201026_VUS_Input'
    sFileTag        = '20201026_VUS_Input_BRCA1'

    #sFileTag        = '20200909_ClinVar'
    sClinVarFile    = '%s/ClinVar/%s_%s.txt'               % (sInputDir, sFileTag, sGenome)

    sPAMIndelFile   = '%s/2020_NBME_PAMIndelFreqTable.txt' % sInputDir
    dict_PAM_Indel  = load_PAM_indel_freq (sPAMIndelFile)

    sTestKey        = 'sub.1'

    ## RefSeq Database ##
    sRefFlat          = '%s/%s_refFlat_step4.txt' % (sGENOME_DIR, sGENOME)
    list_cRefGene     = parse_refflat_line(sRefFlat, 'list', 'Gene')
    dict_cRefGene     = filter_refflat(list_cRefGene)


    nGuideUp        = 24
    nGuideDown      = 73
    nBufferSize     = 100

    nSetPBSLen      = 13   # Limit PBS to set size
    nSetRTLen       = 12   # Limit RT to set size

    nMinPBS         = 0
    nMaxPBS         = 13

    nMaxRT_forPAM   = 12 # Window for PAM
    nMaxRT          = 12 # Max RT Length

    nMaxEditPosWin  = 15 # Distance between PAM and mutation
    nAltBuffer1     = 10
    nAltBuffer2     = 20

    nPegCnt         = 4  # pegRNA per PAM per Round 1:min:min+5, 2:min+11:
    nFinalPegCnt    = 8  # pegRNA per PAM
    list_sAltKey    = ['sub.1', 'del.1', 'ins.1',
                       'del.2', 'ins.2', 'del.3',
                       'sub.2', 'ins.3', 'sub.3']

    dict_sWinSize   = {'substitution':{1:[nMaxRT_forPAM - 1 - 3, 6], 2:[nMaxRT_forPAM - 2 - 3, 6], 3:[nMaxRT_forPAM - 3 - 3, 6]},
                       'insertion':   {1:[nMaxRT_forPAM - 2 - 3, 6], 2:[nMaxRT_forPAM - 3 - 3, 6], 3:[nMaxRT_forPAM - 4 - 3, 6]},
                       'deletion':    {1:[nMaxRT_forPAM - 1 - 3, 6], 2:[nMaxRT_forPAM - 1 - 3, 6], 3:[nMaxRT_forPAM - 1 - 3, 6]}}

    dict_sPAM_Dist  = {'+': {0: [(4, 5, 6), {'AGG':'CGT', 'CGG':'CGT', 'GGG':None, 'TGG':None}],
                             1: [(3, 4, 5), {'GCG': 'GCC', 'GGG': 'GGC', 'CTG': 'CTC', 'TTG': 'CTC',
                                             'CCG': 'CCC', 'CGG': 'CGC', 'AGG': 'CGC', 'TCG': 'TCC',
                                             'ACG': 'ACC', 'GTG': 'GTC'}],
                             2: [None, None]},

                       '-': { 0: [None, None],
                              1: [(5, 6, 7),  {'GGA': 'ACT', 'GGC': 'CGC', 'GGG': None, 'GGT': 'CGT'}],
                              2: [[(3, 4, 5), {'ACG':'CCT', 'CCG':'CCT', 'GCG':'CCT', 'TCG':'TCT'}],
                                  [(7, 8, 9), {'GGC': 'TGC', 'GCC': 'TCC', 'GAT': 'TAT',
                                              'GAG': 'TAG', 'GGG': 'TGG', 'GCG': 'TCG',
                                              'GCT': 'TGA', 'GGA': 'TGA', 'GGT': 'TGT',
                                              'GAC': 'TAC'}]],
                             }
                       }
    list_sPool      = ['Model']
    for sPool in list_sPool:

        sOutDir        = '%s/ClinVar_GuideDesign/%s_VUS2' % (sOutputDir, sPool)
        os.makedirs(sOutDir, exist_ok=True)

        list_cVCF      = determine_sequence(sPool, sClinVarFile, cGenome, list_sAltKey, dict_sWinSize, dict_cRefGene,
                                            nGuideUp, nGuideDown, nBufferSize, nMinPBS, nMaxPBS, nMaxRT,
                                            nSetPBSLen, nSetRTLen, nMaxEditPosWin, bTestRun, sTestKey)


        #dict_sAltKey    = random_selection_PBS_RT (list_cVCF, nPegCnt, nMaxRT, nAltBuffer1, nAltBuffer2, bTestRun, sTestKey)
        dict_sAltKey    = all_selection_PBS_RT (list_cVCF, dict_sPAM_Dist, dict_cRefGene, bTestRun, sTestKey)

        sOutTag         = '201102_Clinvar_flank%s_VUS_BRCA1' % nBufferSize
        #dict_sRandomPEG = generate_random_pegRNAs (sPool, list_sAltKey, nGuideUp, nGuideDown, nMaxPBS, nMaxRT, nMaxEditPosWin, nAltBuffer1, nAltBuffer2, nPegCnt, bTestRun)
        #output_pegRNA (sPool, sOutDir, sOutTag, nFinalPegCnt, dict_sAltKey, dict_sRandomPEG, bTestRun, sTestKey)
        output_pegRNA_nolimit (sPool, sOutDir, sOutTag, dict_sPAM_Dist, dict_cRefGene, dict_sAltKey, dict_PAM_Indel, bTestRun, sTestKey)

    #loop END: sPool
#def END: determine_clinvar_mutations_VUS


def determine_clinvar_mutations (sInputDir, sOutputDir):

    sGenome         = 'hg38'
    sGenomeFile     = '%s/%s.fa' % (sGENOME_DIR, sGenome)
    cGenome         = cFasta(sGenomeFile)

    sFileTag        = '20200421_ClinVar_hg38'
    sClinVarFile    = '%s/ClinVar/%s.txt' % (sInputDir, sFileTag)

    bTestRun        = False
    sTestKey        = 'ins.3'

    nGuideUp        = 24
    nGuideDown      = 73
    nBufferSize     = 100
    nMaxPBS         = 17
    nMaxRT_forPAM   = 40

    nMaxRT          = 40
    nMaxEditPosWin  = 30
    nAltBuffer1     = 10
    nAltBuffer2     = 20


    nPegCnt         = 4  # pegRNA per PAM per Round 1:min:min+5, 2:min+11:
    nFinalPegCnt    = 8  # pegRNA per PAM
    list_sAltKey    = ['sub.1', 'del.1', 'ins.1',
                       'del.2', 'ins.2', 'del.3',
                       'sub.2', 'ins.3', 'sub.3']

    dict_sWinSize   = {'substitution':{1:[nMaxRT_forPAM - 1 - 3, 6], 2:[nMaxRT_forPAM - 2 - 3, 6], 3:[nMaxRT_forPAM - 3 - 3, 6]},
                       'insertion':   {1:[nMaxRT_forPAM - 2 - 3, 6], 2:[nMaxRT_forPAM - 3 - 3, 6], 3:[nMaxRT_forPAM - 4 - 3, 6]},
                       'deletion':    {1:[nMaxRT_forPAM - 1 - 3, 6], 2:[nMaxRT_forPAM - 1 - 3, 6], 3:[nMaxRT_forPAM - 1 - 3, 6]}}


    list_sPool      = ['Model', 'Therapy']
    for sPool in list_sPool:

        sOutDir         = '%s/ClinVar_GuideDesign/%s' % (sOutputDir, sPool)
        os.makedirs(sOutDir, exist_ok=True)

        dict_sRandomPEG = generate_random_pegRNAs (sPool, list_sAltKey, nGuideUp, nGuideDown, nMaxPBS, nMaxRT, nMaxEditPosWin, nAltBuffer1, nAltBuffer2, nPegCnt, bTestRun)

        list_cVCF       = determine_sequence(sPool, sClinVarFile, cGenome, dict_sWinSize, nGuideUp, nGuideDown, nBufferSize, nMaxPBS, nMaxRT, nMaxEditPosWin, bTestRun, sTestKey)

        dict_sAltKey    = random_selection_PBS_RT (list_cVCF, nPegCnt, nMaxRT, nAltBuffer1, nAltBuffer2, bTestRun, sTestKey)

        sOutTag         = '2020612_Clinvar_flank%s' % nBufferSize

        output_pegRNA  (sPool, sOutDir, sOutTag, nFinalPegCnt, dict_sAltKey, dict_sRandomPEG, bTestRun, sTestKey)
    #loop END: sPool
#def END: determine_clinvar_mutations


def generate_random_pegRNAs (sPool, list_sAltKey, nGuideUp, nGuideDown, nMaxPBS, nMaxRT, nMaxEditPosWin,  nAltBuffer1, nAltBuffer2, nPegCnt, bTestRun):

    print('generate_random_pegRNAs')
    nRandomCnt    = 8000 # Arbitrary number
    dict_sOutput  = {}
    list_sNucs    = ['A', 'C', 'G', 'T']

    for sAltKey in list_sAltKey:

        if sAltKey not in dict_sOutput:
            dict_sOutput[sAltKey] = {}

        sAltType, sAltLen = sAltKey.split('.')
        nAltLen           = int(sAltLen)
        sAltType          = {'sub':'substitution', 'del':'deletion', 'ins':'insertion'}[sAltType]

        for i in range(nRandomCnt):

            for sStrand in ['+', '-']:

                sGuideUpSeq     = ''.join((random.choice(list_sNucs) for i in range(nGuideUp)))
                sGuideDownSeq   = ''.join((random.choice(list_sNucs) for i in range(nGuideDown)))

                sPAMSeq        = random.choice(list_sNucs) + 'GG'
                sRefGuideSeq   = sGuideUpSeq + sPAMSeq + sGuideDownSeq
                nPAM_Nick      = nGuideUp - 3
                list_nAltRange = [i for i in range(nGuideUp + len(sPAMSeq), nPAM_Nick + nMaxRT + 1)]
                nAltIndex      = random.choice(list_nAltRange)
                nAltPosWin     = set_alt_position_window_v2 (sPool, sStrand, sAltKey, nAltIndex, nGuideUp, nGuideUp + 3, nAltLen, True)

                ## AltPosWin Filter ##
                if nAltPosWin <= 0:             continue
                if nAltPosWin > nMaxEditPosWin: continue
                #################

                ## Substitution Filter ##
                #if sAltKey.startswith('sub'):
                #    if nAltPosWin < -3: continue
                ##########################

                if sAltType == 'substitution':
                    sRefNuc      = sRefGuideSeq[nAltIndex:nAltIndex + nAltLen]
                    sAltNuc      = ''.join((random.choice(list_sNucs) for i in range(nAltLen)))
                    list_sCheck  = [1 if sRef == sAlt else 0 for sRef, sAlt in zip(sRefNuc, sAltNuc)]
                    list_sNewAlt    = []
                    for i in range(len(list_sCheck)):
                        if list_sCheck[i] == 1:
                            list_sNewNucs = [sNuc for sNuc in list_sNucs if sNuc != sRefNuc[i]]
                            sNewAlt       = ''.join(random.choice(list_sNewNucs))
                            list_sNewAlt.append(sNewAlt)
                        else:
                            list_sNewAlt.append(sAltNuc[i])
                    #loop END: i
                    sNewAltNuc = ''.join(list_sNewAlt)

                    sAltGuideSeq = sRefGuideSeq[:nAltIndex] + sNewAltNuc + sRefGuideSeq[(nAltIndex + len(sAltNuc)):]

                if sAltType == 'deletion':
                    sRefNuc      = sRefGuideSeq[nAltIndex:nAltIndex + nAltLen + 1]
                    sAltNuc      = sRefGuideSeq[nAltIndex]
                    sBufferSeq   = ''.join((random.choice(['A', 'C', 'G', 'T']) for i in range(len(sRefNuc) - 1)))
                    sAltGuideSeq = sRefGuideSeq[:nAltIndex] + sAltNuc + sRefGuideSeq[(nAltIndex + len(sRefNuc)):] + sBufferSeq

                if sAltType == 'insertion':
                    sRefNuc      = sRefGuideSeq[nAltIndex]
                    sAltNuc      = sRefNuc + ''.join((random.choice(['A', 'C', 'G', 'T']) for i in range(nAltLen)))
                    sAltGuideSeq = sRefGuideSeq[:nAltIndex] + sAltNuc  + sRefGuideSeq[(nAltIndex + 1):-nAltLen]

                if sPool == 'Therapy':
                    sGuideSeq = sAltGuideSeq
                    sTempSeq  = sRefGuideSeq
                else:
                    sGuideSeq = sRefGuideSeq
                    sTempSeq  = sAltGuideSeq

                sPAMKey       = '%s,%s,%s,None' % (sStrand, sGuideSeq, nAltPosWin)

                #print(sGuideSeq, sPAMSeq, sStrand, nGuideUp, nGuideUp+3, nPAM_Nick, nAltPosWin, sRefNuc, sAltNuc)

                if sPAMKey not in dict_sOutput[sAltKey]:
                    dict_sOutput[sAltKey][sPAMKey] = []

                dict_sPBS, dict_sRT = {}, {}
                list_nPBSLen = [nNo + 1 for nNo in range(nMaxPBS)]
                for nPBSLen in list_nPBSLen:
                    nPBSStart = nPAM_Nick - nPBSLen  # 5' -> PamNick
                    nPBSEnd = nPAM_Nick
                    sPBSSeq = sTempSeq[nPBSStart:nPBSEnd]
                    if bTestRun: print('>' * nPBSStart + sPBSSeq, nPBSStart, nPBSEnd, len(sPBSSeq))

                    sKey       = len(sPBSSeq)
                    if sKey not in dict_sPBS:
                        dict_sPBS[sKey] = ''
                    dict_sPBS[sKey] = sPBSSeq
                #loop END: nPBSLen

                list_nRTPos = [nNo + 1 for nNo in range(nAltIndex + nAltLen, (nPAM_Nick + nMaxRT))]
                for nRTPos in list_nRTPos:
                    nRTStart   = nPAM_Nick   # PamNick -> 3'
                    nRTEnd     = nRTPos
                    sRTSeq     = sTempSeq[nRTStart:nRTEnd]

                    if bTestRun: print('>' *  nPAM_Nick + sRTSeq, nRTStart, nRTEnd, len(sRTSeq))

                    sKey       = len(sRTSeq)

                    if sKey > nMaxRT: continue

                    if not sRTSeq: continue

                    if sPool == 'Therapy':
                        if sAltKey.startswith('del'):
                            if sKey < nAltPosWin + nAltLen: continue
                    else:
                        if sAltKey.startswith('ins'):
                            if sKey < nAltPosWin + nAltLen: continue
                    #if END:
                    if sKey not in dict_sRT:
                        dict_sRT[sKey] = ''
                    dict_sRT[sKey] = sRTSeq
                #loop END: nRTPos

                list_nPBSKeys   = list(dict_sPBS.keys())
                list_nAllRTKeys = list(dict_sRT.keys())

                dict_nMaxRT = {1: [nAltPosWin, nAltPosWin + nAltBuffer1],  # Round 1
                               2: [nAltPosWin + nAltBuffer1 + 1, nAltPosWin + nAltBuffer2]}  # Round 2


                for sRound in dict_nMaxRT:

                    nMinRTLen, nMaxRTLen = dict_nMaxRT[sRound]
                    if nMinRTLen - 1 > nMaxRT: continue

                    list_nRTKeys         = [nRTKey for nRTKey in list_nAllRTKeys if nMinRTLen <= nRTKey <= nMaxRTLen]

                    if len(list_nRTKeys) < nPegCnt:
                        nAdjustMinRT = nMinRTLen - (nPegCnt - len(list_nRTKeys))
                        list_nRTKeys = [nRTKey for nRTKey in list_nAllRTKeys if nAdjustMinRT <= nRTKey <= nMaxRTLen]

                    if min(len(list_nPBSKeys), len(list_nRTKeys)) < nPegCnt:
                        nMinK = min(len(list_nPBSKeys), len(list_nRTKeys))
                    else:
                        nMinK = nPegCnt

                    list_nPBS_random = np.random.choice(list(dict_sPBS.keys()), nMinK, replace=False)
                    list_nRT_random  = np.random.choice(list(dict_sRT.keys()), nMinK, replace=False)

                    if not dict_sRT: continue

                    for nPBSKey, nRTKey in zip(list_nPBS_random, list_nRT_random):
                        sPBSSeq = dict_sPBS[nPBSKey]
                        sRTSeq  = dict_sRT[nRTKey]

                        sOut    = '%s,%s,%s,%s,%s,%s,%s,%s,%s' % ('None', sStrand, 'None',
                                                                  sAltType, nAltLen, nAltPosWin,
                                                                  sGuideSeq, sPBSSeq, sRTSeq)
                        dict_sOutput[sAltKey][sPAMKey].append(sOut)
                    #loop END: sPBSKey, sRTKey
                #loop END: sRound
            #loop END: sStrand

        #loop END: i
    #loop END: sAltKey

    return dict_sOutput
#def END: generate_random_pegRNAs


def determine_sequence(sPool, sClinVarFile, cGenome, list_sAltKey, dict_sWinSize, dict_cRefGene, nGuideUp, nGuideDown, nBufferSize, nMinPBS, nMaxPBS, nMaxRT, nSetPBSLen, nSetRTLen, nMaxEditPosWin, bTestRun, sTestKey):

    print('determine_sequence', sPool)

    dict_sRE          = {'+': '[ACGT]GG', '-': 'CC[ACGT]'}
    list_cVCF         = load_clinvar_PE(sClinVarFile, nBufferSize, cGenome)
    dict_sORFData     = get_gene_ORFData (list_cVCF, dict_cRefGene, cGenome, 3)
    dict_sStats       = {} # For testing purpose
    dict_sStats2      = {} # For testing purpose

    #for e in dict_sORFData['BRCA1']:
    #    print(e, dict_sORFData['BRCA1'][e])

    nFrame             = 0
    nNoFrame           = 0
    list_cVCF_filtered = []
    dict_sTest         = {}
    for cVCF in list_cVCF:

        cRef          = dict_cRefGene[cVCF.sGeneSym][0]
        sAltKey       = '%s.%s' % (cVCF.sAltType[:3], cVCF.nAltLen)
        nAltIndex     = len(cVCF.sRefSeq) - nBufferSize - 1
        cVCF.sAltSeq  = cVCF.sRefSeq[:nAltIndex] + cVCF.sAltNuc + cVCF.sRefSeq[(nAltIndex + len(cVCF.sRefNuc)):]
        sForGuideSeq, sForTempSeq = set_target_seq(cVCF, sPool)

        if cVCF.sAltNuc  not in dict_sTest:
            dict_sTest[cVCF.sAltNuc] = 0
        dict_sTest[cVCF.sAltNuc] += 1

        if bTestRun:
            if sAltKey   != sTestKey:  continue
            if cVCF.nPos != 43124096: continue
            print(cVCF.sAltType, cVCF.sChrID, cVCF.nPos, cVCF.sRefNuc, cVCF.sAltNuc)
            print(cVCF.sRefSeq)
            print(sForGuideSeq)
            print(sForTempSeq)
            #print(reverse_complement(sForGuideSeq))
            #print(reverse_complement(sForTempSeq))

        #if END: bTestRun

        cVCF.dict_PAM = {}
        for sStrand in ['+', '-']:

            sRE = dict_sRE[sStrand]

            for sReIndex in regex.finditer(sRE, sForGuideSeq, overlapped=True):
                nIndexStart     = sReIndex.start()
                nIndexEnd       = sReIndex.end()
                sPAMSeq         = sForGuideSeq[nIndexStart:nIndexEnd]

                nAltPosWin      = set_alt_position_window_v2 (sPool, sStrand, sAltKey, nAltIndex, nIndexStart, nIndexEnd, cVCF.nAltLen, False)
                nGenomicS_PAM   = cVCF.nStartPos  + nIndexStart
                nGenomicE_PAM   = cVCF.nStartPos  + nIndexEnd - 1

                ## AltPosWin Filter ##
                if nAltPosWin <= 0:             continue
                if nAltPosWin > nMaxEditPosWin: continue

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
                '''
                #################

                nPAM_Nick = set_PAM_nicking_pos (cVCF, sStrand, sPool, nAltIndex, nIndexStart, nIndexEnd)

                if not check_PAM_window(cVCF, dict_sWinSize,
                                        sStrand, nIndexStart,
                                        nIndexEnd, nAltIndex): continue

                if sStrand == '+':
                    sGuideSeq       = sForGuideSeq[nIndexStart - nGuideUp:nIndexEnd + nGuideDown]
                    nGenomicS_Guide = cVCF.nStartPos + nIndexStart - nGuideUp
                    nGenomicE_Guide = cVCF.nStartPos + nIndexEnd + nGuideDown - 1

                else:
                    sGuideSeq       = reverse_complement(sForGuideSeq[nIndexStart - nGuideDown:nIndexEnd + nGuideUp])
                    nGenomicS_Guide = cVCF.nStartPos + nIndexStart - nGuideDown
                    nGenomicE_Guide = cVCF.nStartPos + nIndexEnd + nGuideUp - 1
                #if END: sStrand

                nFrameNo       = determine_PAM_ORF (cVCF, sStrand, dict_sORFData, sPAMSeq, nGenomicS_PAM, nGenomicE_PAM)
                if nFrameNo is None:
                    nNoFrame += 1
                    continue

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
            #loop END: sReIndex
        #loop END: sStrand

        nFrame += len(cVCF.dict_PAM)
        if cVCF.dict_PAM:
            list_cVCF_filtered.append(cVCF)
    #loop END: cVCF
    print('list_cVCF', len(list_cVCF))
    print('list_cVCF_filtered', len(list_cVCF_filtered))

    #if bTestRun: sys.exit()
    #else:        return list_cVCF
    return list_cVCF
#def END: determine_seqs


def get_gene_ORFData (list_cVCF, dict_cRefGene, cGenome, nBufferSize):

    dict_cVCF   = {}
    for cVCF in list_cVCF:
        dict_cVCF.setdefault(cVCF.sGeneSym, []).append(cVCF)
    #loop END: cVCF

    dict_sOutput  = {}
    for sGeneSym  in dict_cVCF:

        #if sGeneSym != 'PCARE':continue
        #if sGeneSym != 'PEX10':continue

        cRef            = dict_cRefGene[sGeneSym][0]
        #print(cRef.sGeneSym, cRef.sChrID, cRef.sStrand)

        obtain_sORFSeq  (cRef, cGenome)

        if sGeneSym not in dict_sOutput:
            dict_sOutput[sGeneSym] = {}

        list_nORFPos = []
        for sExonKey in cRef.dict_nExonSeq:

            nExonS, nExonE = sExonKey
            sExonSeq       = cRef.dict_nExonSeq[sExonKey]
            list_nORFPos  += [(nExonS + i, nNt) for i, nNt in enumerate(sExonSeq)]
        #loop END: sExonKey

        for nFrameNo in [0, 1, 2]:

            for i in range(0, len(list_nORFPos), nBufferSize):

                try: list_nPamPos = list_nORFPos[i+nFrameNo:i+nFrameNo+nBufferSize]
                except IndexError: continue

                if len(list_nPamPos) != nBufferSize: continue

                list_nPos = [nPamPos[0] for nPamPos in list_nPamPos]
                sPAMSeq   = ''.join([nPamPos[1] for nPamPos in list_nPamPos])

                sPosKey   = (list_nPos[0],list_nPos[2])

                if sPosKey not in dict_sOutput[sGeneSym]:
                    dict_sOutput[sGeneSym][sPosKey] = [nFrameNo, sPAMSeq]
            #loop END: i
        #loop END: nFrameNo
    #loop END: sGeneSym

    return dict_sOutput
#def END: get_gene_ORFData


def set_target_seq (cVCF, sPool):

    if sPool == 'Therapy':
        sForGuideSeq = cVCF.sAltSeq
        sForTempSeq  = cVCF.sRefSeq
    else:
        sForGuideSeq = cVCF.sRefSeq
        sForTempSeq  = cVCF.sAltSeq

    return sForGuideSeq, sForTempSeq
#def END: set_target_seq


def set_alt_position_window (sStrand, sAltKey, nAltIndex, nIndexStart, nIndexEnd, nAltLen):

    if sStrand == '+': return nAltIndex - nIndexStart
    else:
        if sAltKey.startswith('sub'):
            return nIndexEnd - nAltIndex - 1 - (nAltLen - 1)
        else:
            return nIndexEnd - 1  - nAltIndex - 1
#def END: set_alt_position_window


def set_alt_position_window_v2 (sPool, sStrand, sAltKey, nAltIndex, nIndexStart, nIndexEnd, nAltLen, bRandom):

    if bRandom:
        if sStrand == '+':
            if sAltKey.startswith('sub'):   return (nAltIndex + 1) - (nIndexStart - 3)
            else:                           return (nAltIndex + 1) - (nIndexStart - 3) + 1
        else:
            if sAltKey.startswith('sub'):   return (nAltIndex + 1) - (nIndexStart - 3)
            else:                           return (nAltIndex + 1) - (nIndexStart - 3) + 1

    else:

        if sStrand == '+':
            if sAltKey.startswith('sub'):   return (nAltIndex + 1) - (nIndexStart - 3)
            else:                           return (nAltIndex + 1) - (nIndexStart - 3) + 1
        else:
            if sAltKey.startswith('sub'):
                return nIndexEnd - nAltIndex + 3 - (nAltLen - 1)

            elif sAltKey.startswith('del'):
                if sPool == 'Therapy': return nIndexEnd - nAltIndex + 3
                else:                  return nIndexEnd - nAltIndex + 3 - nAltLen

            else:
                if sPool == 'Therapy': return nIndexEnd - nAltIndex + 3 - nAltLen
                else:                  return nIndexEnd - nAltIndex + 3
            #if END:
        #if END:
    #if END:
#def END: set_alt_position_window


def set_PAM_nicking_pos(cVCF, sStrand, sPool, nAltIndex, nIndexStart, nIndexEnd):

    if sStrand == '-':
        #if nIndexEnd <= nAltIndex:
        if cVCF.sAltType == 'deletion':
            if sPool == 'Therapy':
                nPAM_Nick = nIndexEnd + 3 + cVCF.nAltLen
            else:
                nPAM_Nick = nIndexEnd + 3 - cVCF.nAltLen

        elif cVCF.sAltType == 'insertion':
            if sPool == 'Therapy':
                nPAM_Nick = nIndexEnd + 3 - cVCF.nAltLen
            else:
                nPAM_Nick = nIndexEnd + 3 + cVCF.nAltLen

        elif cVCF.sAltType == 'substitution':
            nPAM_Nick = nIndexEnd + 3
    else:
        if nIndexStart >= nAltIndex:
            if cVCF.sAltType == 'deletion':
                if sPool == 'Therapy':
                    nPAM_Nick = nIndexStart - 3
                else:
                    nPAM_Nick = nIndexStart - 3

            elif cVCF.sAltType == 'insertion':
                if sPool == 'Therapy':
                    nPAM_Nick = nIndexStart - 3
                else:
                    nPAM_Nick = nIndexStart - 3

            elif cVCF.sAltType == 'substitution':
                nPAM_Nick = nIndexStart - 3

        else:
            nPAM_Nick = nIndexStart - 3
        #if END: nIndexStart
    #if END: sStrand
    return nPAM_Nick
#def END: set_PAM_Nicking_Pos


def check_PAM_window (cVCF, dict_sWinSize, sStrand, nIndexStart, nIndexEnd, nAltIndex):

    nUp, nDown = dict_sWinSize[cVCF.sAltType][cVCF.nAltLen]

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


def determine_PAM_ORF (cVCF, sStrand, dict_sORFData, sPAMSeq, nGenomicS_PAM, nGenomicE_PAM):
    try:
        sORFCheck = dict_sORFData[cVCF.sGeneSym][(nGenomicS_PAM, nGenomicE_PAM)]
        nFrameNo, sPAMCheck = sORFCheck

        if sStrand == '-':
            if nFrameNo == 1: nFrameNo = 2
            if nFrameNo == 2: nFrameNo = 1

        if sPAMSeq == sPAMCheck:  return nFrameNo
    except KeyError: return None
#def END: determine_PAM_ORF


def PAM_pos_in_guide_check (cVCF, sStrand, sGuideSeq, sPAM, sPAMSeq, sForGuideSeq, sForTempSeq):

    sCheckPAM = reverse_complement(sPAMSeq) if sStrand == '-' else sPAMSeq

    if sPAM != sCheckPAM:
        print('PAMCheck Failed')
        print(sStrand, cVCF.sAltType, cVCF.sChrID, cVCF.nPos, cVCF.sRefNuc, cVCF.sAltNuc)
        print(sForGuideSeq)
        print(sForTempSeq)
        print(sGuideSeq)
        print('revcom', reverse_complement(sGuideSeq))
        print(sCheckPAM)
        print(sPAM)
        sys.exit()
#def END: PAM_pos_in_guide_check


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


def random_selection_PBS_RT (list_cVCF, nPegCnt, nMaxRT, nAltBuffer1, nAltBuffer2, bTestRun, sTestKey):

    dict_K          = {'sub.1': nPegCnt, 'sub.2': nPegCnt, 'sub.3': nPegCnt,
                       'del.1': nPegCnt, 'del.2': nPegCnt, 'del.3': nPegCnt,
                       'ins.1': nPegCnt, 'ins.2': nPegCnt, 'ins.3': nPegCnt}

    dict_nWeights   = {17: [0.02] * 5 + [0.075] * 12,  # Selection Rate Probability
                       16: [0.02] * 5 + [0.08] * 10 + [0.10],
                       15: [0.02] * 5 + [0.09] * 10, }
    dict_sAltKey    = {}
    dict_sStats     = {} # For testing purpose
    for cVCF in list_cVCF:

        sAltKey = '%s.%s' % (cVCF.sAltType[:3], cVCF.nAltLen)
        nK      = dict_K[sAltKey]

        if bTestRun:
            if sAltKey != sTestKey: continue
            #if cVCF.nPos != 114713976: continue
            pass

        if sAltKey not in dict_sAltKey:
            dict_sAltKey[sAltKey] = {}

        for sPAMKey in cVCF.dict_PAM:

            if sPAMKey not in dict_sAltKey[sAltKey]:
                dict_sAltKey[sAltKey][sPAMKey] = []

            dict_sPBS, dict_sRT = cVCF.dict_PAM[sPAMKey]
            list_nPBSKeys       = list(dict_sPBS.keys())
            list_nAllRTKeys     = list(dict_sRT.keys())
            sStrand             = sPAMKey.split(',')[0]
            sGuideSeq           = sPAMKey.split(',')[1]
            nAltPosWin          = int(sPAMKey.split(',')[2])
            dict_nMaxRT         = {1: [nAltPosWin, nAltPosWin + nAltBuffer1],                    # Round 1
                                   2: [nAltPosWin + nAltBuffer1 + 1, nAltPosWin + nAltBuffer2]}  # Round 2

            for sRound in dict_nMaxRT:

                nMinRTLen, nMaxRTLen = dict_nMaxRT[sRound]
                if nMinRTLen - 1 > nMaxRT: continue

                list_nRTKeys         = [nRTKey for nRTKey in list_nAllRTKeys if nMinRTLen <= nRTKey <= nMaxRTLen]

                if len(list_nRTKeys) < nPegCnt:
                    nAdjustMinRT = nMinRTLen - (nPegCnt - len(list_nRTKeys))
                    list_nRTKeys = [nRTKey for nRTKey in list_nAllRTKeys if nAdjustMinRT <= nRTKey <= nMaxRTLen]

                if min(len(list_nPBSKeys), len(list_nRTKeys)) < nPegCnt:
                    nMinK = min(len(list_nPBSKeys), len(list_nRTKeys))
                else:
                    nMinK = nPegCnt

                list_nWeights_PBS = dict_nWeights[len(list_nPBSKeys)]
                list_nPBS_random  = np.random.choice(list_nPBSKeys, nMinK, p=list_nWeights_PBS, replace=False)
                list_nRT_random   = np.random.choice(list_nRTKeys, nMinK, replace=False)

                for nPBSKey, nRTKey in zip(list_nPBS_random, list_nRT_random):

                    if nRTKey not in dict_sStats:
                        dict_sStats[nRTKey] = 0
                    dict_sStats[nRTKey] += 1

                    sPBSSeq = dict_sPBS[nPBSKey]
                    sRTSeq  = dict_sRT[nRTKey]
                    sOut    = '%s,%s,%s,%s,%s,%s,%s,%s,%s' % (cVCF.sChrID, sStrand, cVCF.nPos,
                                                              cVCF.sAltType, cVCF.nAltLen, nAltPosWin,
                                                              sGuideSeq, sPBSSeq, sRTSeq)

                    dict_sAltKey[sAltKey][sPAMKey].append(sOut)
                #loop END: sPBSKey, sRTKey
            #loop END: sRound
        #loop END: sPAMKey
    #loop END: cVCF

    return dict_sAltKey
#def END: random_selection_PBS_RT


def all_selection_PBS_RT (list_cVCF, dict_sPAM_Dist, dict_cRefGene, bTestRun, sTestKey):

    dict_sAltKey = {}
    dict_sStats  = {} # For testing purpose
    dict_sStats2 = {} # For testing purpose
    nPAMNoAlt    = 0
    nPAMAlt      = 0

    for cVCF in list_cVCF:

        sAltKey = '%s.%s' % (cVCF.sAltType[:3], cVCF.nAltLen)
        cRef    = dict_cRefGene[cVCF.sGeneSym][0]
        if bTestRun:
            if sAltKey != sTestKey: continue
            if cVCF.nPos != 43124096: continue
            pass

        if sAltKey not in dict_sAltKey:
            dict_sAltKey[sAltKey] = {}

        for sPAMKey in cVCF.dict_PAM:

            dict_sPBS, dict_sRT = cVCF.dict_PAM[sPAMKey]
            list_nPBSKeys       = list(dict_sPBS.keys())
            list_nRTKeys        = list(dict_sRT.keys())
            sStrand             = sPAMKey.split(',')[0]
            sGuideSeq           = sPAMKey.split(',')[1]
            nAltPosWin          = int(sPAMKey.split(',')[2])
            sAltNuc             = sPAMKey.split(',')[3]
            sPAMSeq             = sPAMKey.split(',')[4]
            nFrame              = int(sPAMKey.split(',')[5])
            sGeneSym            = sPAMKey.split(',')[6]

            sPBSSeq, sRTSeq     = dict_sPBS[list_nPBSKeys[0]], dict_sRT[list_nRTKeys[0]]

            #print('1', sPBSSeq, sRTSeq)

            if len(dict_sPBS) != 1: continue
            if len(dict_sRT)  != 1: continue

            if nAltPosWin in [5, 6]:
                nFrame     = 'N'
                nPAMNoAlt += 1
            else:
                list_sSeq = make_PAM_dist(cRef.sStrand, sStrand, nFrame, dict_sPAM_Dist, sPBSSeq, sRTSeq)

                if list_sSeq is None:  continue
                else:
                    sPBSSeq, sRTSeq = list_sSeq
                    nPAMAlt += 1
            #if END:
            #print('2', sPBSSeq, sRTSeq)

            sOut    = '%s,%s,%s,%s,%s,%s,Frame%s,%s,%s,%s' % (cVCF.sChrID, sStrand, cVCF.nPos,
                                                              cVCF.sAltType, cVCF.nAltLen, nAltPosWin,
                                                              nFrame, sGuideSeq, sPBSSeq, sRTSeq)

            if sPAMKey not in dict_sAltKey[sAltKey]:
                dict_sAltKey[sAltKey][sPAMKey] = ''

            dict_sAltKey[sAltKey][sPAMKey] = sOut

            if sAltKey not in dict_sStats:
                dict_sStats[sAltKey] = 0
            dict_sStats[sAltKey] += 1

            #loop END: sPBSKey, sRTKey
        #loop END: sPAMKey
    #loop END: cVCF

    #for sAltKey in dict_sStats:
    #    print(sAltKey, dict_sStats[sAltKey])

    return dict_sAltKey
#def END: all_selection_PBS_RT


def make_PAM_dist (sGeneStrand, sPAMStrand, nFrame, dict_sPAM_Dist, sPBSSeq, sRTSeq):

    sKey = '%s%s' % (sPAMStrand, nFrame)
    if sKey != '-2':

        if sGeneStrand == sPAMStrand:
            list_nTarPos, dict_sTarSeq = dict_sPAM_Dist['+'][nFrame]
        else:
            list_nTarPos, dict_sTarSeq = dict_sPAM_Dist[sPAMStrand][nFrame]

        if list_nTarPos is None:   return None

        elif dict_sTarSeq is None: return None

        else:

            list_nTarPos = [i - 1 for i in list_nTarPos]  # Adjust for 0-based indexes
            sTarSeq      = ''.join([sRTSeq[i] for i in list_nTarPos])

            #print('sTarSeq', sTarSeq)

            try: sAltSeq      = dict_sTarSeq[sTarSeq]
            except KeyError: sAltSeq = None

            #print('sAltSeq', sAltSeq)

            if sAltSeq is None: return None
            else:
                dict_sAltIndex = {nPos:sAltSeq[i] for i, nPos in enumerate(list_nTarPos)}
                sAltRTSeq      = ''.join([dict_sAltIndex[i] if i in list_nTarPos else sNT for i, sNT in enumerate(sRTSeq)])

                #print('sAltRTSeq',sAltRTSeq)

                return sPBSSeq, sAltRTSeq
            #if END:
        #if END:
    else:

        for list_nTarPos, dict_sTarSeq in dict_sPAM_Dist[sStrand][nFrame]:

            list_nTarPos = [i - 1 for i in list_nTarPos]  # Adjust for 0-based indexes
            sTarSeq      = ''.join([sRTSeq[i] for i in list_nTarPos])

            try: sAltSeq = dict_sTarSeq[sTarSeq]
            except KeyError: continue

            if sAltSeq is None: return None
            else:
                dict_sAltIndex = {nPos: sAltSeq[i] for i, nPos in enumerate(list_nTarPos)}
                sAltRTSeq = ''.join([dict_sAltIndex[i] if i in list_nTarPos else sNT for i, sNT in enumerate(sRTSeq)])
                return sPBSSeq, sAltRTSeq
            #if END:
        #loop END: list_nTarPos, dict_sTarSeq
    #if END:
#def END: make_PAM_dist


def output_pegRNA (sPool, sOutputDir, sOutTag, nFinalPegCnt, dict_sAltKey, dict_sRandomPEG, bTestRun, sTestKey):
    dict_pegRNACnt  = {'Therapy':{'sub.1':39500, 'del.1':16000, 'ins.1':9600,
                                  'del.2':6400,  'ins.2':2400,  'del.3':2000,
                                  'sub.2':1200,  'ins.3':1200,  'sub.3':1200},
                       'Model':  {'sub.1':9000,  'del.1':4000,  'ins.1':2400,
                                  'del.2':1600,  'ins.2':1000,  'del.3':500,
                                  'sub.2':500,   'ins.3':500,   'sub.3':500}}
    dict_nPosCheck  = {}
    dict_nPosCheck2 = {}
    dict_sSeedCnt   = {}
    dict_sPegRNACnt = {}
    for sAltKey in dict_sAltKey:

        if bTestRun:
            if sAltKey != sTestKey: continue

        sOutFile           = '%s/%s.%s.output.%s.txt' % (sOutputDir, sOutTag, sPool, sAltKey)
        OutFile            = open(sOutFile, 'w')

        list_sOutput       = [sPAMKey for sPAMKey in dict_sAltKey[sAltKey] if len(dict_sAltKey[sAltKey][sPAMKey]) == nFinalPegCnt]
        list_sRandomPAMs   = [sPAMKey for sPAMKey in dict_sRandomPEG[sAltKey] if len(dict_sRandomPEG[sAltKey][sPAMKey]) == nFinalPegCnt]
        nFinalCnt          = dict_pegRNACnt[sPool][sAltKey]

        print(sAltKey, len(list_sOutput))


        if len(list_sOutput) < nFinalCnt:
            list_sFinalOutput = list_sOutput + random.sample(list_sRandomPAMs, (nFinalCnt - len(list_sOutput)))
        else:
            list_sFinalOutput = random.sample(list_sOutput, nFinalCnt)
        #if END:
        print(sAltKey, len(list_sFinalOutput))


        if sAltKey not in dict_sSeedCnt:
            dict_sSeedCnt[sAltKey] = 0
        dict_sSeedCnt[sAltKey] = len(list_sFinalOutput)

        if sAltKey not in dict_nPosCheck:
            dict_nPosCheck[sAltKey] = {}

        if sAltKey not in dict_sPegRNACnt:
            dict_sPegRNACnt[sAltKey] = 0

        for sPAMKey in list_sFinalOutput:

            sStrand, sGuideSeq, nAltPosWin, nPos = sPAMKey.split(',')

            nAltPosWin = int(nAltPosWin)

            if nAltPosWin not in dict_nPosCheck[sAltKey]:
                dict_nPosCheck[sAltKey][nAltPosWin] = 0
            dict_nPosCheck[sAltKey][nAltPosWin] += 1

            if nAltPosWin not in dict_nPosCheck2:
                dict_nPosCheck2[nAltPosWin] = 0
            dict_nPosCheck2[nAltPosWin] += 1
            try:
                for sOut in dict_sAltKey[sAltKey][sPAMKey]:
                    OutFile.write('%s\n' % sOut)
                    dict_sPegRNACnt[sAltKey] += 1
            except KeyError:
                for sOut in dict_sRandomPEG[sAltKey][sPAMKey]:
                    OutFile.write('%s\n' % sOut)
                    dict_sPegRNACnt[sAltKey] += 1
            #end TRY:
        #loop END: sOut
        OutFile.close()
    #loop END: sAltKey
    if bTestRun:
        list_sAltKey = [sTestKey]
    else:
        list_sAltKey = ['sub.1', 'del.1', 'ins.1', 'del.2', 'ins.2', 'del.3', 'sub.2', 'ins.3', 'sub.3']
    print('sAltKey', 'SeedGuides', 'pegRNAs')

    for sAltKey in list_sAltKey:
        print(sAltKey, dict_sSeedCnt[sAltKey], dict_sPegRNACnt[sAltKey])

    for nAltPosWin in dict_nPosCheck2:
        print(nAltPosWin, dict_nPosCheck2[nAltPosWin])

    #for sAltKey in dict_nPosCheck:
    #    print(sAltKey)
    #    list_nPos = sorted(list(dict_nPosCheck[sAltKey].keys()))
    #    for nPos in list_nPos:
    #        print(nPos, dict_nPosCheck[sAltKey][nPos])
#def END: output_pegRNA


def output_pegRNA_nolimit (sPool, sOutputDir, sOutTag, dict_sPAM_Dist, dict_cRefGene, dict_sAltKey, dict_PAM_Indel, bTestRun, sTestKey):

    dict_nPosCheck  = {}
    dict_nPosCheck2 = {}
    dict_sPegRNACnt = {}
    dict_sPAMCheck  = {}
    for sAltKey in dict_sAltKey:

        if bTestRun:
            if sAltKey != sTestKey: continue

        if sAltKey not in dict_nPosCheck:
            dict_nPosCheck[sAltKey] = {}

        if sAltKey not in dict_sPegRNACnt:
            dict_sPegRNACnt[sAltKey] = 0

        sOutFile           = '%s/%s.%s.output.PAMDist.%s.txt' % (sOutputDir, sOutTag, sPool, sAltKey)
        OutFile            = open(sOutFile, 'w')

        list_sOutput       = [sPAMKey for sPAMKey in dict_sAltKey[sAltKey]]

        for sPAMKey in list_sOutput:

            sStrand       = sPAMKey.split(',')[0]
            sGuideSeq     = sPAMKey.split(',')[1]
            nAltPosWin    = int(sPAMKey.split(',')[2])
            sAltNuc       = sPAMKey.split(',')[3]
            sPAMSeq       = sPAMKey.split(',')[4]
            nFrame        = int(sPAMKey.split(',')[5])
            sGeneSym      = sPAMKey.split(',')[6]
            cRef          = dict_cRefGene[sGeneSym][0]

            if nAltPosWin not in dict_nPosCheck[sAltKey]:
                dict_nPosCheck[sAltKey][nAltPosWin] = 0
            dict_nPosCheck[sAltKey][nAltPosWin] += 1

            if nAltPosWin not in dict_nPosCheck2:
                dict_nPosCheck2[nAltPosWin] = 0
            dict_nPosCheck2[nAltPosWin] += 1

            sOut       = dict_sAltKey[sAltKey][sPAMKey]
            sPBSSeq    = sOut.split(',')[-2]
            sRTSeq     = sOut.split(',')[-1]

            sPAM_Alt   = sRTSeq[3:6]
            fIndelFreq = dict_PAM_Indel[sPAM_Alt]

            if fIndelFreq > 5:

                list_sSeq = make_PAM_dist (cRef.sStrand, sStrand, nFrame, dict_sPAM_Dist, sPBSSeq, sRTSeq)

                if list_sSeq is None: continue

                sNewRTSeq   = list_sSeq[1]
                sPAM_Alt2   = sNewRTSeq[3:6]
                fIndelFreq2 = dict_PAM_Indel[sPAM_Alt2]
                sNewOut     = ','.join([sData for sData in sOut.split(',')[:-1]] + [sNewRTSeq])

                if sPAM_Alt == sPAM_Alt2: continue  ## Skip those PAM that can't be altered again

                #print('Re PAM Dist')
                #print(sPAMSeq, sPAM_Alt, sPAM_Alt2, nFrame)
                #print(sOut, fIndelFreq, fIndelFreq2)
                #print(sRTSeq)
                #print(sNewRTSeq)

                OutFile.write('%s,%s,%s\n' % (sNewOut, fIndelFreq2, sGeneSym))
                dict_sPegRNACnt[sAltKey] += 1
            else:

                #print('No PAM Dist')
                #print(sPAMSeq, sPAM_Alt, sRTSeq, nFrame)
                #print(sOut, fIndelFreq)


                OutFile.write('%s,%s,%s\n' % (sOut, fIndelFreq, sGeneSym))
                dict_sPegRNACnt[sAltKey] += 1


                #if nAltPosWin in [4,7]:
                #    print(sPAMSeq, sPAM_Alt, sRTSeq)
                #    print(sOut, fIndelFreq)
            #loop END: sOut
        #loop END: sPAMKey

        OutFile.close()
    #loop END: sAltKey

    for sKey in dict_sPAMCheck:
        print(sKey, dict_sPAMCheck[sKey])

    #if bTestRun: list_sAltKey = [sTestKey]
    #else:        list_sAltKey = ['sub.1', 'del.1', 'ins.1', 'del.2', 'ins.2', 'del.3', 'sub.2', 'ins.3', 'sub.3']

    #for sAltKey in list_sAltKey:
    #    print(sAltKey, dict_sPegRNACnt[sAltKey])

    #for nAltPosWin in dict_nPosCheck2:
    #    print(nAltPosWin, dict_nPosCheck2[nAltPosWin])

    #for sAltKey in dict_nPosCheck:
    #    print(sAltKey)
    #    list_nPos = sorted(list(dict_nPosCheck[sAltKey].keys()))
    #    for nPos in list_nPos:
    #        print(nPos, dict_nPosCheck[sAltKey][nPos])
#def END: output_pegRNA


def cVCF_parse_vcf_files_clinvar(sVCFFile):
    if not os.path.isfile(sVCFFile):
        sys.exit('File Not Found %s' % sVCFFile)

    list_sOutput = []
    InFile = open(sVCFFile, 'r')

    for sReadLine in InFile:
        # File Format
        # Column Number:     | 0       | 1        | 2          | 3       | 4
        # Column Description:| sChrID  | nPos     | sDBSNP_ID  | sRefNuc | sAltNuc
        # Column Example:    | 1       | 32906558 | rs79483201 | T       | A
        # Column Number:     | 5       | 6        | 7          | 8              | 9./..
        # Column Description:| fQual   | sFilter  | sInfo      | sFormat        | sSampleIDs
        # Column Example:    | 5645.6  | PASS     | .          | GT:AD:DP:GQ:PL | Scores corresponding to sFormat

        if sReadLine.startswith('#'): continue  # SKIP Information Headers
        list_sColumn = sReadLine.strip('\n').split('\t')

        cVCF        = cVCFData()
        cVCF.sChrID = 'chr%s' % list_sColumn[0]

        if list_sColumn[0].startswith('MT'): continue
        if list_sColumn[0].startswith('NW'): continue

        try:
            cVCF.nPos = int(list_sColumn[1])
        except ValueError:
            continue

        cVCF.sDBSNP_ID  = list_sColumn[2]
        cVCF.sRefNuc    = list_sColumn[3]
        cVCF.sAltNuc    = list_sColumn[4]
        cVCF.fQual      = float(list_sColumn[5]) if list_sColumn[5] != '.' else list_sColumn[5]
        cVCF.sFilter    = list_sColumn[6]
        cVCF.sInfo      = list_sColumn[7]
        cVCF.dict_sInfo = dict([sInfo.split('=') for sInfo in cVCF.sInfo.split(';') if len(sInfo.split('=')) == 2])

        # try: cVCF.sAlleleFreq    = float(dict_sInfo['AF_raw'])
        # except ValueError: cVCF.sAlleleFreq = np.mean([float(f) for f in  dict_sInfo['AF_raw'].split(',')])

        list_sOutput.append(cVCF)
    #loop END: sReadLine
    InFile.close()

    return list_sOutput
#def END: cVCF_parse_vcf_files_clinvar


def positive_control (sInputDir, sOutputDir):

    ## HGNC Database ##
    sHGNC_File        = '%s/HGNC_genelist.txt' % sREF_DIR
    dict_sHGNC        = load_HGNC_reference (sHGNC_File)

    ## RefSeq Database ##
    sRefFlat          = '%s/%s_refFlat_step4.txt' % (sGENOME_DIR, sGENOME)
    dict_cRefGene     = parse_refflat_line(sRefFlat, 'dict', 'Gene')

    ## Genome FASTA ##
    sGenomeFile       = '%s/%s/%s.fa' % (sREF_DIR, sGENOME, sGENOME)
    cGenome           = cFasta(sGenomeFile)

    sTargetEssentials = '%s/TargetGenes_PosCtrl.txt' % sInputDir
    dict_sTargetGenes = load_target_genes_posctrl (sTargetEssentials)
    print('Target Genes', len(dict_sTargetGenes))

    get_guides_from_genome (sOutputDir, dict_sHGNC, dict_cRefGene, cGenome, dict_sTargetGenes)

#def END: positive_control


def VUS_liftover_BRCA1_data (sInputDir):

    sLiftDir = '%s/LiftOver_VUS'  % sInputDir
    sFileTag = '20201026_VUS_Input_BRCA1'
    run_liftover_VUS(sLiftDir, sFileTag)

    dict_sLiftover   = {}
    sLiftoverBedfile = '%s/%s_hg38.bed' % (sLiftDir, sFileTag)
    InFile          = open(sLiftoverBedfile, 'r')
    for sReadLine in InFile:

        list_sColumn = sReadLine.strip('\n').split('\t')
        sChrID, nPos, nPosPlus1, sAAchange = list_sColumn

        if sAAchange not in dict_sLiftover:
            dict_sLiftover[sAAchange] = 0
        dict_sLiftover[sAAchange] = nPos
    #loop END: sReadLine
    InFile.close()

    sInFile  = '%s/%s_hg19.txt' % (sLiftDir, sFileTag)
    sOutFile = '%s/ClinVar/%s_hg38.txt' % (sInputDir, sFileTag)

    InFile   = open(sInFile, 'r')
    OutFile  = open(sOutFile, 'w')

    for sReadLine in InFile:

        list_sColumn = sReadLine.strip('\n').split('\t')

        nChrNo       = list_sColumn[0]
        nOldPos      = list_sColumn[1]
        sAAchange    = list_sColumn[2]
        sRef         = reverse_complement(list_sColumn[3])
        sAlt         = reverse_complement(list_sColumn[4])
        sAltSize     = list_sColumn[5]
        sAltType     = list_sColumn[6]
        sGeneSym     = list_sColumn[7]

        nNewPos      = dict_sLiftover[sAAchange]

        sOut = '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' % (nChrNo, nNewPos, sAAchange, sRef, sAlt,
                                                     sAltSize, sAltType, sGeneSym)

        OutFile.write(sOut)
    #loop END: sReadLine
    InFile.close()
    OutFile.close()
#def END: VUS_liftover_BRCA1_data


def temp_anal (sInputDir):

    sInFile       = '%s/20201214_FullGuideSeqInfo.txt' % sInputDir
    InFile        = open(sInFile, 'r')
    dict_sBarcode = {}
    nCnt          = 0
    for sReadLine in InFile:

        if sReadLine.startswith('Index'): continue

        nCnt += 1

        nIndex        = sReadLine.strip('\n').split(',')[0]
        sBarcode      = sReadLine.strip('\n').split(',')[1]
        sOldGuidse     = sReadLine.strip('\n').split(',')[2]
        sNewGuide     = sReadLine.strip('\n').split(',')[3].upper()

        sKey          = sNewGuide

        if sKey not in dict_sBarcode:
            dict_sBarcode[sKey] = ''
        dict_sBarcode[sKey] = sBarcode
    #loop END: sReadLine
    InFile.close()

    print('nCnt 1', nCnt)

    ## Cross-check and filter final guide output with all labels
    sInFile      = '%s/20201214_UPDATED_PE_CV_BC list.csv' % sInputDir
    InFile       = open(sInFile, 'r')
    dict_sOutput = {}

    for sReadLine in InFile:

        if sReadLine.startswith('IndexKey'): continue

        sGuideKey     = sReadLine.strip('\n').split(',')[0].upper()
        sExtBarcode   = sReadLine.strip('\n').split(',')[1]
        sRefSeq       = sReadLine.strip('\n').split(',')[2]
        sWTTarSeq     = sReadLine.strip('\n').split(',')[3]
        sEditedTarSeq = sReadLine.strip('\n').split(',')[4]

        try: sBarcode      = dict_sBarcode[sGuideKey]
        except KeyError: continue

        sOut = '%s,%s,%s,%s,%s' % (sBarcode, sExtBarcode, sRefSeq, sWTTarSeq, sEditedTarSeq)

        if sGuideKey not in dict_sOutput:
            dict_sOutput[sGuideKey] = ''
        dict_sOutput[sGuideKey] = sOut
    #loop END: sReadLine
    InFile.close()

    list_sKey = sorted(dict_sOutput.keys(), key=lambda e:int(e.split('_')[-1]))

    sOutFile  = '%s/20201214_FullGuideInfo.txt' % sInputDir
    OutFile   = open(sOutFile, 'w')

    for sKey in list_sKey:

        sOut = '%s,%s\n' % (sKey, dict_sOutput[sKey])
        OutFile.write(sOut)
    #loop END: sKey
    OutFile.close()
#def END: temp_anal


def temp_anal_v2 (sInputDir):

    sInFile       = '%s/20201214_FullGuideSeqInfo.txt' % sInputDir
    InFile        = open(sInFile, 'r')
    dict_sBarcode = {}
    nCnt          = 0
    for sReadLine in InFile:

        if sReadLine.startswith('Index'): continue
        #Index,Barcode,RefSeq, TargetSeq
        nCnt += 1

        sGuideKey   = sReadLine.strip('\n').split(',')[0]
        sBarcode    = sReadLine.strip('\n').split(',')[1]
        sExtBarcode = sReadLine.strip('\n').split(',')[2]
        sRefSeq     = sReadLine.strip('\n').split(',')[3]
        sWTSeq      = sReadLine.strip('\n').split(',')[4]
        sEditedSeq  = sReadLine.strip('\n').split(',')[5]

        sKey        = sGuideKey

        if sKey not in dict_sBarcode:
            dict_sBarcode[sKey] = ''
        dict_sBarcode[sKey] = '%s,%s,%s' % (sExtBarcode, sWTSeq, sEditedSeq)
    #loop END: sReadLine
    InFile.close()

    print('nCnt 1', nCnt)

    ## Make Guide Key, Barcode and EditPos File
    sInFile      = '%s/20201214_GuideKey_EditPos.csv' % sInputDir
    InFile       = open(sInFile, 'r')
    dict_sOutput = {}

    for sReadLine in InFile:

        if sReadLine.startswith('INDEX'): continue
        #INDEX,Edit Position,PBS length,RTT length
        sGuideKey     = sReadLine.strip('\n').split(',')[0].upper()
        sEditPos      = sReadLine.strip('\n').split(',')[1]
        sPBSLen       = sReadLine.strip('\n').split(',')[2]
        sRTLen        = sReadLine.strip('\n').split(',')[3]

        try: sExtBarcode, sWTSeq, sEditedSeq      = dict_sBarcode[sGuideKey].split(',')
        except KeyError: continue

        sOut = '%s,%s,%s,%s,%s,%s' % (sExtBarcode, sEditPos, sPBSLen, sRTLen, sWTSeq, sEditedSeq)

        if sGuideKey not in dict_sOutput:
            dict_sOutput[sGuideKey] = ''
        dict_sOutput[sGuideKey] = sOut
    #loop END: sReadLine
    InFile.close()

    list_sKey = sorted(dict_sOutput.keys(), key=lambda e:int(e.split('_')[-1]))

    sOutFile  = '%s/20210113_GuideKeyEditExtBarcode.txt' % sInputDir
    OutFile   = open(sOutFile, 'w')

    for sKey in list_sKey:

        sOut = '%s,%s\n' % (sKey, dict_sOutput[sKey])
        OutFile.write(sOut)
    #loop END: sKey
    OutFile.close()
#def END: temp_anal_v2


def others_distribution (sAnalysis, sInputDir, sOutputDir):

    ## Guide Run List: Profiling, ClinVar
    #list_sGuideRun  = ['Profiling', 'ClinVar']
    list_sGuideRun  = ['ClinVar']

    ## Error Type: ErrorProne, ErrorFree
    #list_sErrorType = ['ErrorFree', 'ErrorProne']
    list_sErrorType = ['ErrorFree']

    sFileList   = '%s/FileLists/%s.txt'  % (sInputDir, sAnalysis)
    dict_sFiles = load_NGS_files (sFileList)
    nNickIndex  = 19

    bSUBOnly    = 1

    nLenCutoff  = 52 # most common wt/edited length


    for sRun in list_sGuideRun:

        sInFile        = '%s/20210113_%s_GuideKeyEditExtBarcode.txt' % (sInputDir, sRun)
        dict_sGuideKey = load_guidekey (sInFile, bSUBOnly)

        for sError in list_sErrorType:
            sRunName      = '%s_%s' % (sRun, sError)
            sInDir        = '%s/%s_%s_%s' % (sOutputDir, sAnalysis, sRun, sError)
            list_sSamples = list(dict_sFiles.keys())

            for sSample in list_sSamples:
                print('Processing %s %s - %s' % (sAnalysis, sRunName, sSample))

                sInFile       = '%s/%s.OtherReads.txt' % (sInDir, sSample)
                dict_sOthers  = load_other_reads (sInFile)
                list_sOthers  = list(dict_sOthers.keys())
                dict_sDistro  = {}
                dict_sTarLen  = {}
                dict_s23Check = {}
                nDiffLenCnt   = 0
                nMaxCnt       = 0

                for sBarcodeKey in list_sOthers:
                    #sBarcodeKey = 'CACTCTTTTTTTCTATGACTATCATCACGCGGGGCT'  #DEL3
                    #sBarcodeKey = 'TTCTTATTTTTTGACTACTCGTACATACATACCACT'  #INS3

                    #sBarcodeKey = 'TGCTGGTTTTTTTCTAGCTCACTCGACGTATCCTGG'  #SUB1
                    #sBarcodeKey = 'GGGGTCTTTTTTTATGCACACTCTCTCTCATGAGGG'  #SUB2
                    #sBarcodeKey = 'TGCCGTTTTTTTATCAGCTCGTACTCTATGAGTTGG'  #SUB3

                    #sBarcodeKey = 'TGACGCTTTTTTAGTATCACTCTACTATGCGCATCT'

                    try: cGuide = dict_sGuideKey[sBarcodeKey]
                    except KeyError: continue

                    if sBarcodeKey not in dict_sDistro:
                        dict_sDistro[sBarcodeKey] = {'sub':{}, 'indel':{}}

                    ## WT Length Distribution ##

                    #if len(cGuide.sWTSeq) not in dict_sTarLen:
                        #dict_sTarLen[len(cGuide.sWTSeq)] = 0
                    #dict_sTarLen[len(cGuide.sWTSeq)] += 1

                    #if len(cGuide.sWTSeq) != nLenCutoff: continue
                    ##############################

                    list_sOtherReads = list(dict_sOthers[sBarcodeKey].keys())

                    for i, sOtherRead in enumerate(list_sOtherReads):
                        nMaxCnt += 1
                        if len(sOtherRead) != len(cGuide.sEditedSeq):
                            nDiffLenCnt += 1
                            continue

                        nOtherCnt = int(dict_sOthers[sBarcodeKey][sOtherRead])
                        check_others (cGuide, nNickIndex, sOtherRead, nOtherCnt, dict_sDistro[sBarcodeKey], dict_s23Check)
                    #loop END: sOtherRead
                #loop END: sBarcodeKey

                for sType in dict_s23Check:
                    print(sType, len(dict_s23Check[sType]))
                    list_sBarcodes = list(dict_s23Check[sType].keys())

                    for sBarcodeKey in list_sBarcodes[:200]:

                        if dict_s23Check[sType][sBarcodeKey]:
                            print('-----', sBarcodeKey)

                        for sSeq1, sSeq2, sSeq3, sSeq4 in dict_s23Check[sType][sBarcodeKey]:
                            print(sSeq1)
                            print(sSeq2)
                            print(sSeq3)
                            print(sSeq4)
                        #loop END: sSeq1, sSeq2, sSeq3, sSeq4
                    #loop END: sBarcodeKey
                    sys.exit()
                #loop END: sType
                sys.exit()

                print('nDiffLenCnt', nDiffLenCnt)
                print('nMaxCnt', nMaxCnt)

                dict_sFullDist  = {}
                for sBarcodeKey in dict_sDistro:
                    for sType in dict_sDistro[sBarcodeKey]:

                        if sType not in dict_sFullDist:
                            dict_sFullDist[sType] = {}

                        for nIndex in dict_sDistro[sBarcodeKey][sType]:

                            nCnt = dict_sDistro[sBarcodeKey][sType][nIndex]
                            #nCnt = 1

                            if nIndex not in dict_sFullDist[sType]:
                                dict_sFullDist[sType][nIndex] = 0
                            dict_sFullDist[sType][nIndex] += nCnt
                        #loop END: nIndex
                    #loop END: sType
                #loop END: sBarcodeKey

                nMaxinSub   = max([nIndex for nIndex in dict_sFullDist['sub']])
                nMaxinIndel = max([nIndex for nIndex in dict_sFullDist['indel']])

                print('sub', nMaxinSub)
                print('indel', nMaxinIndel)

                list_nDistSub   = [0 for i in range(nMaxinSub+1)]
                list_nDistIndel = [0 for i in range(nMaxinIndel+1)]


                for sType in ['sub', 'indel']:

                    for nIndex in dict_sFullDist[sType]:
                        if sType == 'sub': list_nDistSub[nIndex]   += dict_sFullDist[sType][nIndex]
                        else:              list_nDistIndel[nIndex] += dict_sFullDist[sType][nIndex]
                    #loop END: nIndex

                print(list_nDistSub)
                print(list_nDistIndel)
                sys.exit()
            #loop END: sSample
        #loop END: sError
    #loop END: sRun
#def END: others_distribution


def load_guidekey (sInFile, bSUBOnly):

    InFile       = open(sInFile, 'r')
    dict_sOutput = {}
    list_sTest   = []
    for sReadLine in InFile:
        #MODELING_DEL1PEGRNA_1,CTCCACTTTTTTAGTGATCTCGATAGATACGTGTTT,4,11,14,GTTTTGGCGTGGAGGGAGGTCCAGGGGTGAAAATCA,GTTTTGGCGTGGAGGGAGGTCCGGGGTGAAAATCA
        list_sColumns      = sReadLine.strip('\n').split(',')
        cGuide             = cGuideKeyInfo()
        cGuide.sGuideKey   = list_sColumns[0].upper()
        cGuide.sLibrary    = cGuide.sGuideKey.split('_')[0]
        cGuide.sRunType    = cGuide.sGuideKey.split('_')[1].replace('PEGRNA', '')
        cGuide.nMutLen     = int(cGuide.sRunType[-1])
        cGuide.sExtBarcode = list_sColumns[1]
        cGuide.nEditPos    = int(list_sColumns[2])
        cGuide.nPBSLen     = int(list_sColumns[3])
        cGuide.sRTLen      = int(list_sColumns[4])
        cGuide.sWTSeq      = list_sColumns[5]
        cGuide.sEditedSeq  = list_sColumns[6]

        list_sTest.append(len(cGuide.sEditedSeq))


        if bSUBOnly:
            if not cGuide.sRunType.startswith('SUB'): continue


        if cGuide.sExtBarcode not in dict_sOutput:
            dict_sOutput[cGuide.sExtBarcode] = ''
        dict_sOutput[cGuide.sExtBarcode] = cGuide

    #loop END: sReadLine

    print('Max Edited', max(list_sTest))
    print('Min Edited', min(list_sTest))

    InFile.close()
    return dict_sOutput
#def END: load_guidekey


def load_other_reads (sInFile):

    InFile       = open(sInFile, 'r')
    dict_sOutput = {}

    for sReadLine in InFile:

        list_sColumns = sReadLine.strip('\n').split('\t')

        sBarcodeKey   = list_sColumns[0]

        if not list_sColumns[1]: continue
        dict_sOthers  = dict([sAltSeqCnt.split(':') for sAltSeqCnt in list_sColumns[1].split(',')])

        if sBarcodeKey not in dict_sOutput:
            dict_sOutput[sBarcodeKey] = ''
        dict_sOutput[sBarcodeKey] = dict_sOthers
    #loop END: sReadLine
    InFile.close()

    return dict_sOutput
#def END: load_other_reads


def check_others (cGuide, nNickIndex, sOtherRead, nOtherCnt, dict_sDistro, dict_s23Check):
    list_sWT = []
    list_sAT = []
    list_sOT = []

    #for nIndex in range(nNickIndex, len(cGuide.sWTSeq)):
    for nIndex in range(len(cGuide.sWTSeq)):
        if 'INS' in cGuide.sRunType:
            list_sWT.append(cGuide.sWTSeq[nIndex] if nIndex != (nNickIndex + cGuide.nEditPos - 1) else '_' * cGuide.nMutLen + cGuide.sWTSeq[nIndex])
        else: list_sWT.append(cGuide.sWTSeq[nIndex])
    #loop END: nIndex

    #for nIndex in range(nNickIndex, len(cGuide.sEditedSeq)):
    for nIndex in range(len(cGuide.sEditedSeq)):
        if 'DEL' in cGuide.sRunType:
            list_sAT.append(cGuide.sEditedSeq[nIndex] if nIndex != (nNickIndex + cGuide.nEditPos - 1) else '_' * cGuide.nMutLen + cGuide.sEditedSeq[nIndex])
        else:
            list_sAT.append(cGuide.sEditedSeq[nIndex])
        list_sOT.append(sOtherRead[nIndex])
    #loop END: nIndex


    sCheckSeq_WT = ''.join([sWT.lower() if i in [(nNickIndex + cGuide.nEditPos-1) + i for i in range(cGuide.nMutLen)]  else sWT for i, sWT in enumerate(list_sWT)])
    sCheckSeq_AT = ''.join([sAT.lower() if i in [(nNickIndex + cGuide.nEditPos-1) + i for i in range(cGuide.nMutLen)]  else sAT for i, sAT in enumerate(list_sAT)])
    sCheckSeq_OT = ''.join([sOT.lower() if i in [(nNickIndex + cGuide.nEditPos-1) + i for i in range(cGuide.nMutLen)]  else sOT for i, sOT in enumerate(list_sOT)])

    list_sMM = []
    for nIndex in range(len(sCheckSeq_OT)):
        #if not (sCheckSeq_WT[nIndex] == sCheckSeq_AT[nIndex] == sCheckSeq_OT[nIndex]):
        if not (sCheckSeq_WT[nIndex] == sCheckSeq_OT[nIndex]):
            bFlag = 1
        else:
            bFlag = 0
        list_sMM.append(bFlag)
    #loop END: nIndex
    sMMNotation = ''.join([str(i) for i in list_sMM])

    bFlag     = ''
    nCntCheck = 0
    for i in list_sMM:
        if i == 1: nCntCheck += i
        else:      nCntCheck = 0
        if nCntCheck >= 3:
            bFlag = 'indel'
            break
        else: bFlag = 'sub'
    #loop END: i

    if bFlag == 'indel':  list_nHits = [min([i for i in range(len(list_sMM)) if list_sMM[i] == 1])]
    else:                 list_nHits = [i for i in range(len(list_sMM)) if list_sMM[i] == 1]

    for nHitIndex in list_nHits:
        if nHitIndex not in dict_sDistro[bFlag]:
            dict_sDistro[bFlag][nHitIndex] = 0
        dict_sDistro[bFlag][nHitIndex] += nOtherCnt
    #loop END: nHitIndex

    sCheck1 = sCheckSeq_WT, sCheckSeq_WT[nNickIndex+cGuide.nEditPos-1:nNickIndex+cGuide.nEditPos+cGuide.nMutLen-1]
    sCheck2 = sCheckSeq_AT, sCheckSeq_AT[nNickIndex+cGuide.nEditPos-1:nNickIndex+cGuide.nEditPos+cGuide.nMutLen-1]
    sCheck3 = sCheckSeq_OT, sCheckSeq_OT[nNickIndex+cGuide.nEditPos-1:nNickIndex+cGuide.nEditPos+cGuide.nMutLen-1]
    sCheck4 = sMMNotation.replace('0', ' ').replace('1', '*')

    sKey1   = cGuide.sRunType
    sKey2   = cGuide.sExtBarcode

    if sKey1 not in dict_s23Check:
        dict_s23Check[sKey1] = {}
    if sKey2 not in dict_s23Check[sKey1]:
        dict_s23Check[sKey1][sKey2] = []

    if 23 in list_nHits:
        dict_s23Check[sKey1][sKey2].append([sCheckSeq_WT, sCheckSeq_AT, sCheckSeq_OT, sCheck4])
#def END: check_EditWin


def main():

    bTestRun = False

    ## 10/31/2019 Prime Editing KO Screening w/ HKK and Goosang ##
    sAnalysis   = 'HKK_201214' # PE Full

    sInputDir   = '%s/pe_screening/input'  % sDATA_DIR
    sOutputDir  = '%s/pe_screening/output' % sDATA_DIR

    ## Clinvar Data ##
    VUS_basicstat_makeinput (sInputDir, sOutputDir)

    #VUS_library_input_get_genomic_position (sInputDir)
    #VUS_liftover_BRCA1_data (sInputDir)

    determine_clinvar_mutations (sInputDir, sOutputDir)
    determine_clinvar_mutations_VUS (sInputDir, sOutputDir, bTestRun)

    ## Target Gene List for Positive Control 20201006 ##
    #positive_control (sInputDir, sOutputDir)

    ## TEMP ##
    #temp_anal(sInputDir)
    #temp_anal_v2(sInputDir)

    ## Others Distribution ##
    #others_distribution (sAnalysis, sInputDir, sOutputDir)


    '''

    ## Determine WT and Edited Distribution ##
    #sFileList   = '%s/%s.txt'   % (sInputDir, sAnalysis)
    #dict_sFiles = load_NGS_files (sFileList)
    sFastqTag    = '191212_56K_Plasmid'
    #sFastqTag   = '200116_EditedTest_73'
    #sFastqTag   = 'TEST'
    sInFile      = '%s/fastq/%s.fastq'   % (sInputDir, sFastqTag)
    sFreqOut     = '%s/%s.Freqs.txt'     % (sOutputDir, sFastqTag)
    sReadIDOut   = '%s/%s.readIDs.txt'   % (sOutputDir, sFastqTag)


    ## Load Input Data ##
    sInFile = '%s/Barcode_Targets.txt' % sInputDir
    list_cData = load_PE_input(sInFile)

    ## Analyze Output ##
    list_sData, dict_sReadID = analyze_PE_KO_output (sFreqOut, sReadIDOut)

    nIndex      = 2
    sType       = {1:'WT', 2:'Alt', 3:'Other'}[nIndex]
    list_sData  = sorted(list_sData, key=lambda e:e[nIndex], reverse=True)
    for sData in list_sData[:20]:
        sBarcode, nWT, nEdited, nOther = sData
        list_sReadID = dict_sReadID[sBarcode][sType]
        print(sBarcode, nWT, nEdited, nOther)
        #print(sType, list_sReadID)
    #loop END: sData

    ## MISC Checking available cellline data ##
    #basic_stat_ABSOLUTE_data (sInputDir)

    ## Target Gene List ##
    sTargetEssentials = '%s/TargetGenes.txt' % sInputDir
    #dict_sTargetGenes = load_target_genes (sTargetEssentials)
    #get_guides_from_genome (sOutputDir, dict_sHGNC, dict_cRefGene, cGenome, dict_sTargetGenes)

    ## Short Analyses ##
    sWorkDir = '%s/short_analyses'            % sInputDir
    sOutDir  = '%s/short_analyses'            % sOutputDir

    sInFile  = '%s/candidates_v2.txt'          % sWorkDir
    sOutFile = '%s/candidates_top10_min20.txt' % sOutDir
    list_sOutput = rank_candidates_by_gene (sInFile)

    OutFile  = open(sOutFile, 'w')
    for list_sData in list_sOutput:
        sOut = ','.join(str(sData) for sData in list_sData)
        OutFile.write('%s\n' % sOut)
    #loop END: list_sData
    OutFile.close()
    '''
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
    #if END: len(sys.argv)
#if END: __name__