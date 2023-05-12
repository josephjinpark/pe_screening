#!/usr/bin/env python


import os, sys, pickle, time, subprocess, json, re, string, regex, random, collections, itertools
import subprocess as sp
import numpy as np
import scipy.stats as stats
import matplotlib as mpl
import multiprocessing as mp
from Bio import SeqIO

mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import Locator
from CRISPResso2 import CRISPRessoCOREResources as cp2
from CRISPResso2 import CRISPResso2Align as cp2_align

sGENOME     = 'hg38'
sSRC_DIR    = '/home/hkim/src/pe_screening'
sBASE_DIR   = '/extdata1/Jinman/pe_screening'
sDATA_DIR   = '/extdata1/Jinman'
sEDNAFULL   = '%s/EDNAFULL' % sBASE_DIR
ALN_MATRIX  = cp2_align.read_matrix(sEDNAFULL)

sREF_DIR    = '/data/reference_genome'
sGENOME_DIR = '%s/%s'       % (sREF_DIR, sGENOME)
sCHRSEQ_DIR = '%s/Splited'  % sGENOME_DIR
sTIME_STAMP = '%s'          % (time.ctime().replace(' ', '-').replace(':', '_'))
sLIFTOVER   = '%s/liftOver' % sREF_DIR
sFLASH      = '/home/hkim/bin/FLASH-1.2.11-Linux-x86_64/flash'


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

nLINE_CNT_LIMIT = 1000000 ## split files, must be multiples of 4. file size < 1G recommendation:40000, size > 1G recommendation:400000
nFILE_CNT       = 40
##region Colors

class cColors:
    PURPLE      = '\033[95m'
    CYAN        = '\033[96m'
    DARKCYAN    = '\033[36m'
    BLUE        = '\033[94m'
    GREEN       = '\033[92m'
    YELLOW      = '\033[93m'
    RED         = '\033[91m'
    BOLD        = '\033[1m'
    UNDERLINE   = '\033[4m'
    END         = '\033[0m'


def red(string, e=0):     return '\033[%s31m%s\033[m' % ('' if e == 0 else '1;', string)
def green(string, e=0):   return '\033[%s32m%s\033[m' % ('' if e == 0 else '1;', string)
def yellow(string, e=0):  return '\033[%s33m%s\033[m' % ('' if e == 0 else '1;', string)
def blue(string, e=0):    return '\033[%s34m%s\033[m' % ('' if e == 0 else '1;', string)
def magenta(string, e=0): return '\033[%s35m%s\033[m' % ('' if e == 0 else '1;', string)
def cyan(string, e=0):    return '\033[%s36m%s\033[m' % ('' if e == 0 else '1;', string)
def white(string, e=0):   return '\033[%s37m%s\033[m' % ('' if e == 0 else '1;', string)


def get_color(cMir, sResidue):
    if sResidue == cMir.sAltNuc:
        sResidue = red(sResidue, 1)
    elif sResidue == cMir.sRefNuc:
        sResidue = green(sResidue, 1)
    else:
        sResidue = blue(sResidue, 1)
    return sResidue


#def END: get_color
##endregion

## region Classes

class cAbsoluteCNV: pass
class cGUIDESData: pass
class cPEData:pass


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

        cVCF.sDBSNP_ID = list_sColumn[2]
        cVCF.sRefNuc = list_sColumn[3]
        cVCF.sAltNuc = list_sColumn[4]
        cVCF.fQual = float(list_sColumn[5]) if list_sColumn[5] != '.' else list_sColumn[5]
        cVCF.sFilter = list_sColumn[6]
        cVCF.sInfo = list_sColumn[7]

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


#def END: parse_vcf_stdout


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

        cVCF = cVCFData()
        cVCF.sChrID = 'chr%s' % list_sColumn[0]

        if list_sColumn[0].startswith('MT'): continue
        if list_sColumn[0].startswith('NW'): continue

        try:
            cVCF.nPos = int(list_sColumn[1])
        except ValueError:
            continue

        cVCF.sDBSNP_ID = list_sColumn[2]
        cVCF.sRefNuc = list_sColumn[3]
        cVCF.sAltNuc = list_sColumn[4]
        cVCF.fQual = float(list_sColumn[5]) if list_sColumn[5] != '.' else list_sColumn[5]
        cVCF.sFilter = list_sColumn[6]
        cVCF.sInfo = list_sColumn[7]

        dict_sInfo = dict([sInfo.split('=') for sInfo in cVCF.sInfo.split(';') if len(sInfo.split('=')) == 2])

        # try: cVCF.sAlleleFreq    = float(dict_sInfo['AF_raw'])
        # except ValueError: cVCF.sAlleleFreq = np.mean([float(f) for f in  dict_sInfo['AF_raw'].split(',')])

        list_sOutput.append(cVCF)
    #loop END: sReadLine
    InFile.close()

    return list_sOutput
#def END: cVCF_parse_vcf_files_clinvar


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

        if sStrand == '+':
            return sFetchedSeq

        elif sStrand == '-':
            return reverse_complement(sFetchedSeq)

        else:
            sys.exit('Error: invalid strand')
        #if END: sStrand
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
            nMaxID = max(list_sNMIDs)

            for cRef in dict_cRef[sGene]:

                sKey = int(cRef.sNMID.replace('NM_', ''))

                if sKey == nMaxID: list_sOutput.append(cRef)
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

def obtain_sORFSeq_UTRs(cRef, sChromoSeq):
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
    list_nExonS     = cRef.list_nExonS
    list_nExonE     = cRef.list_nExonE

    cRef.n5UTRStartList  = []
    cRef.n5UTREndList    = [cRef.nORFStartPos]  # For '+' Stand, coding start pos is the end

    cRef.nORFStartList   = [cRef.nORFStartPos]
    cRef.nORFEndList     = [cRef.nORFEndPos]

    cRef.n3UTRStartList  = [cRef.nORFEndPos]  # For '-' Stand, coding end pos is the start
    cRef.n3UTREndList    = []

    # Divide up the exon start positions into three lists 5', ORF, and 3'
    for nStartPos in list_nExonS:
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
    for nEndPos in list_nExonE:
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

def copy_temp_core_script():
    sWorkDir = '%s/src_temp' % sSRC_DIR
    os.makedirs(sWorkDir, exist_ok=True)
    os.system('cp %s/B_main_B206.py %s/tmp_script_%s.py' % (sSRC_DIR, sWorkDir, sTIME_STAMP))
    return '%s/tmp_script_%s.py' % (sWorkDir, sTIME_STAMP)
#def END: copy_temp_core_script


def make_log_dir (sJobName):
    sLogDir = '%s/log/%s/%s' % (sBASE_DIR, sTIME_STAMP, sJobName)
    os.makedirs(sLogDir, exist_ok=True)
    return sLogDir
#def END: make_log_dir


def make_tmp_dir (sJobName):
    sTmpDir = '%s/tmp/%s/%s' % (sBASE_DIR, sTIME_STAMP, sJobName)
    os.makedirs(sTmpDir, exist_ok=True)
    return sTmpDir
#def END: make_tmp_dir


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
    FigHeight       = 10

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


def groupby_element (list_sInput):

    dict_sOutput = {}

    for sInput in list_sInput:

        if sInput not in dict_sOutput:
            dict_sOutput[sInput] = 0
        dict_sOutput[sInput] += 1
    #loop END: sInput

    return dict_sOutput
#def END: groupby_element


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
    list_sTest   = []
    for sReadLine in InFile:
        ## File Format ##
        ## Target#  | Barcode | WT_Target | Edit_Target
        ## 181      | TTT.... | CTGCC..   | CTGCC...

        if sReadLine.startswith('#'): continue ## SKIP HEADER

        list_sColumn = sReadLine.strip('\n').split('\t')

        cPE              = cPEData()
        cPE.sBarcode     = list_sColumn[0]
        cPE.sRefSeq      = list_sColumn[1]
        cPE.sWTSeq       = list_sColumn[2].upper()
        cPE.sAltSeq      = list_sColumn[3].upper()
        sKey             = cPE.sBarcode

        list_sTest.append(cPE.sBarcode)

        if sKey not in dict_sOutput:
            dict_sOutput[sKey] = ''
        dict_sOutput[sKey] = cPE
    #loop END:
    InFile.close()
    '''
    
    print('list_sTest', len(list(set(list_sTest))))

    list_sDup  = [item for item, count in collections.Counter(list_sTest).items() if count > 1]

    print(len(list_sDup))
    for sDup in list_sDup:
        print(sDup)
    sys.exit()
    '''

    return dict_sOutput
#def END: load_PE_input


def load_PE_input_v2 (sInFile):
    dict_sOutput = {}
    InFile       = open(sInFile, 'r')
    list_sTest   = []
    for sReadLine in InFile:
        ## File Format ##
        ## Target#  | Barcode | WT_Target | Edit_Target
        ## 181      | TTT.... | CTGCC..   | CTGCC...

        if sReadLine.endswith('edit\n'): continue ## SKIP HEADER
        if sReadLine.endswith('#\n'): continue ## SKIP HEADER
        if sReadLine.startswith('#'): continue ## SKIP HEADER

        if ',' in sReadLine:
            list_sColumn = sReadLine.strip('\n').split(',')
        else:
            list_sColumn = sReadLine.strip('\n').split('\t')

        cPE              = cPEData()
        cPE.sBarcode     = list_sColumn[0]
        cPE.sRefSeq      = list_sColumn[1].upper().replace('N','')
        cPE.sWTSeq       = list_sColumn[2].upper()
        cPE.sAltSeq      = list_sColumn[3].upper()

        if len(list_sColumn) == 6:
            cPE.sIntendedOnly = list_sColumn[4].upper()
            cPE.sMisMatchOnly = list_sColumn[5].upper()

        sKey             = cPE.sBarcode

        list_sTest.append(cPE.sBarcode)

        if sKey not in dict_sOutput:
            dict_sOutput[sKey] = ''
        dict_sOutput[sKey] = cPE
    #loop END:
    InFile.close()

    return dict_sOutput
#def END: load_PE_input_v2


def load_PE_input_v3 (sInFile):
    dict_sOutput = {}
    InFile       = open(sInFile, 'r')
    list_sTest   = []
    for sReadLine in InFile:
        ## File Format ##
        ## Target#  | Barcode | WT_Target | Edit_Target
        ## 181      | TTT.... | CTGCC..   | CTGCC...

        if sReadLine.endswith('edit\n'): continue ## SKIP HEADER
        if sReadLine.endswith('#\n'): continue ## SKIP HEADER
        if sReadLine.startswith('#'): continue ## SKIP HEADER

        if ',' in sReadLine:
            list_sColumn = sReadLine.strip('\n').split(',')
        else:
            list_sColumn = sReadLine.strip('\n').split('\t')

        cPE              = cPEData()
        cPE.sBarcode     = list_sColumn[0]
        cPE.sRefSeq_conv = list_sColumn[1].upper().replace('N','')
        cPE.sRefSeq_opti = list_sColumn[2].upper().replace('N','')
        cPE.sWTSeq       = list_sColumn[3].upper()
        cPE.sAltSeq      = list_sColumn[4].upper()


        if len(list_sColumn) == 6:
            cPE.sIntendedOnly = list_sColumn[4].upper()
            cPE.sMisMatchOnly = list_sColumn[5].upper()

        sKey             = cPE.sBarcode

        list_sTest.append(cPE.sBarcode)

        if sKey not in dict_sOutput:
            dict_sOutput[sKey] = ''
        dict_sOutput[sKey] = cPE
    #loop END:
    InFile.close()

    return dict_sOutput
#def END: load_PE_input_v3


def rename_samples (sDataDir, sInFile):

    InFile       = open(sInFile, 'r')
    list_sFiles  = [sReadLine.strip('\n').split('\t') for sReadLine in InFile if not sReadLine.startswith('#')]
    dict_sFiles  = dict(list_sFiles)
    InFile.close()

    for sFile in dict_sFiles:

        sOriDir  = '%s/%s'   % (sDataDir, sFile)
        sRawDir  = '%s/raw' % sDataDir
        os.makedirs(sRawDir, exist_ok=True)

        sOriFastq1   = '%s/%s_1.fq.gz' % (sOriDir, sFile.replace('Sample_', ''))
        sOriFastq2   = '%s/%s_2.fq.gz' % (sOriDir, sFile.replace('Sample_', ''))
        sNewFastq1   = '%s/%s_1.fq.gz' % (sRawDir, dict_sFiles[sFile])
        sNewFastq2   = '%s/%s_2.fq.gz' % (sRawDir, dict_sFiles[sFile])

        sCmd        = 'cp -v %s %s; cp -v %s %s;' % (sOriFastq1, sNewFastq1, sOriFastq2, sNewFastq2)
        os.system(sCmd)

    #loop END: sFile
#def END: rename_samples


def load_NGS_files (sAnalysis, filename):

    dict_out = {}
    dict_out_off = {}

    if sAnalysis.startswith('ENDO'):

        infile       = open(filename, 'r', encoding='utf-8-sig')
        for line in infile:

            if line.startswith('Sample'): continue
            if line.startswith('#'): continue
            if '#N/A' in line:continue
            list_col = line.strip('\n').split(',')


            if filename.split('.')[0].endswith('Step1'):

                sample   = list_col[0]
                type     = list_col[1] #rep1, rep2, background
                fastq1   = list_col[2]
                fastq2   = list_col[3]
                barcode1  = list_col[4]
                barcode2  = list_col[5]
                amplicon  = list_col[6]

                key = '%s_%s' % (sample, type)

                if key not in dict_out:
                    dict_out[key] = []
                dict_out[key] = [amplicon, fastq1, fastq2, barcode1, barcode2]


            elif filename.split('.')[0].endswith('OFF'):

                sample    = list_col[0]
                type      = list_col[1] #rep1, rep2, background
                key       = list_col[2]
                fastq1    = list_col[3]
                fastq2    = list_col[4]
                barcode   = list_col[5]
                wtseq     = list_col[6]
                amplicon  = list_col[7]

                altwtseq1    = list_col[8]
                #barcode_prev = list_col[8]

                key  = sample
                key2 = '%s_%s' % (sample, type)

                if key not in dict_out:
                    dict_out[key] = {}

                if key2 not in dict_out[key]:
                    dict_out[key][key2] = []
                dict_out[key][key2] = [barcode, fastq1, fastq2, wtseq, amplicon, altwtseq1]

                if key2 not in dict_out_off:
                    dict_out_off[key2] = []
                dict_out_off[key2] = [barcode, fastq1, fastq2, wtseq, amplicon, altwtseq1]

            else:
                sample   = list_col[0]
                type     = list_col[1] #rep1, rep2, background
                fastq1   = list_col[2]
                fastq2   = list_col[3]
                barcode  = list_col[4]
                wtseq    = list_col[5]
                edseq    = list_col[6]
                amplicon = list_col[7]
                nickpos  = int(list_col[8]) #Nick - 5  Set by HKK 2022-11-16
                rttend   = int(list_col[9]) #RTT + 5

                if len(list_col) > 10:
                    try:nick_ngrna = int(list_col[10])  # RTT + 5
                    except ValueError: nick_ngrna = 'NA'
                else: nick_ngrna = 'NA'
                if filename.split('.')[0].endswith('CTRL'):
                    fastqtag = fastq1.split('_')[0]
                    key = '%s_%s_%s' % (sample, type, fastqtag)
                else:
                    key = '%s_%s' % (sample, type)

                if key not in dict_out:
                    dict_out[key] = []
                dict_out[key] = [barcode, fastq1, fastq2, wtseq, edseq, amplicon, nickpos, rttend, nick_ngrna]

        #loop END: line
        infile.close()

    else:
        InFile       = open(filename, 'r')
        list_sOutput = [sReadLine.strip('\n') for sReadLine in InFile if not sReadLine.startswith('#')]
        InFile.close()

        for sFile in list_sOutput:

            sFileName = sFile.split('.')[0]
            nSampleNo = int(sFileName.split('_')[-1])
            sSample   = '_'.join(sFileName.split('_')[:-1])

            if sSample not in dict_out:
                dict_out[sSample] = {}
            if nSampleNo not in dict_out[sSample]:
                dict_out[sSample][nSampleNo] = ''
            dict_out[sSample][nSampleNo] = sFile
        #loop END: sFile
    #if END:

    return dict_out, dict_out_off
#def END: load_NGS_filesdict_run


def run_FLASH (sAnalysis, sWorkDir, dict_run, dict_off, bTestRun):

    sInDir   = '%s/raw'                 % sWorkDir
    sLogDir  = '%s/log/JJP.RunFLASH.%s' % (sDATA_DIR, sAnalysis)
    os.makedirs(sLogDir, exist_ok=True)

    if sAnalysis.endswith('OFF'): dict_sFiles = dict_off
    else: dict_sFiles = dict_run

    list_sProjects = []
    for sSample in dict_sFiles:
        sOutDir    = '%s/flash/%s'      % (sWorkDir, sSample)
        os.makedirs(sOutDir, exist_ok=True)

        sLogFile   = '%s/%s.fastq.log'  % (sLogDir, sSample)
        sInFile1   = '%s/%s' % (sInDir, dict_sFiles[sSample][1])
        sInFile2   = '%s/%s' % (sInDir, dict_sFiles[sSample][2])

        print(sInFile1, sInFile2)
        assert os.path.isfile(sInFile1) and os.path.isfile(sInFile2)

        sScript      = '%s %s %s ' % (sFLASH, sInFile1, sInFile2)
        sScript     += '-M 400 '   # max overlap
        sScript     += '-m 5 '     # min overlap
        sScript     += '-O '       # allow "outies"  Read ends overlap
                                   # Read 1: <-----------
                                   # Read 2:       ------------>

        sScript     += '-d %s '    % sOutDir
        sScript     += '-o %s '    % sSample

        sCmd         = '%s | tee %s ;'                       % (sScript, sLogFile)
        sCmd        += 'rm -rf %s/*hist* %s/*notCombined* ;' % (sOutDir, sOutDir)

        sFileName    = '%s.others' % sSample
        list_sProjects.append([sSample, sFileName])

        if bTestRun: print(sCmd)
        else: os.system(sCmd)
    #loop END: sSample

    #VS Check
    if not list_sProjects: sys.exit('Empty List : run_FLASH : list_sProjects= %s' % len(list_sProjects))

    sOutFile = '%s/Project_list_%s.txt' % (sWorkDir, sAnalysis)
    OutFile  = open(sOutFile, 'w')
    for sProject, sFileName in list_sProjects:
        sOut = '%s-PE\t%s\n' % (sProject, sFileName)
        OutFile.write(sOut)
    #loop END: sProject, sFileName
#def END: run_FLASH


def split_fq_file (sWorkDir, sFastqTag, bTestRun):

    sOutDir    = '%s/split'                  % sWorkDir
    os.makedirs(sOutDir,exist_ok=True)

    sInFile    = '%s/%s.fastq'               % (sWorkDir, sFastqTag)
    print(sInFile)
    assert os.path.isfile(sInFile)

    sOutTag    = '%s/%s_fastq'               % (sOutDir, sFastqTag)

    sScript     = 'split --verbose '         # For Logging Purposes
    sScript    += '-l %s '                   % nLINE_CNT_LIMIT
    sScript    += '-a 4 '                    # Number of suffice places e.g. 0001.fq = 4

    sScript    += '--numeric-suffixes=1 '    # Start with number 1
    sScript    += '--additional-suffix=.fq ' # Add suffix .fq'
    sScript    += '%s %s_'                   % (sInFile, sOutTag)

    if bTestRun: print(sScript)
    else:
        os.system(sScript)
        list_sFiles = os.listdir(sOutDir)
        sOutFile    = '%s/%s.split.txt'       % (sWorkDir, sFastqTag)
        OutFile     = open(sOutFile, 'w')
        for sFile in list_sFiles:
            if not sFile.endswith('fq'): continue
            sOut = '%s\n' % sFile
            OutFile.write(sOut)
        #loop END: sFile
        OutFile.close()
    #if END:
#def END: split_fq_file


def split_fq_file_v2 (sWorkDir, sFastqTag, bTestRun):

    sOutDir     = '%s/split'                  % sWorkDir
    os.makedirs(sOutDir,exist_ok=True)

    sInFile     = '%s/%s.fastq'               % (sWorkDir, sFastqTag)
    print(sInFile)
    assert os.path.isfile(sInFile)

    sOutTag     = '%s/%s_fastq'               % (sOutDir, sFastqTag)
    sScript     = 'split --verbose '         # For Logging Purposes
    sScript    += '-n %s '                   % nFILE_CNT
    sScript    += '-a 4 '                    # Number of suffice places e.g. 0001.fq = 4

    sScript    += '--numeric-suffixes=1 '    # Start with number 1
    sScript    += '--additional-suffix=.fq ' # Add suffix .fq'
    sScript    += '%s %s_'                   % (sInFile, sOutTag)

    if bTestRun: print(sScript)
    else:
        os.system(sScript)
        list_sFiles = os.listdir(sOutDir)
        sOutFile    = '%s/%s.split.txt'       % (sWorkDir, sFastqTag)
        OutFile     = open(sOutFile, 'w')
        for sFile in list_sFiles:
            sOut = '%s\n' % sFile
            OutFile.write(sOut)
        #loop END: sFile
        OutFile.close()
    #if END:
#def END: split_fq_file


def get_line_cnt (sWorkDir, sFastqTag):

    sInFile  = '%s/%s.fastq'       % (sWorkDir, sFastqTag)
    sOutFile = '%s/%s.linecnt.txt' % (sWorkDir, sFastqTag)

    if os.path.isfile(sOutFile):
        InFile   = open(sOutFile, 'r')
        nLineCnt = int([sReadLine.strip('\n').split(' ') for sReadLine in InFile][0][0])
        InFile.close()
        print('Fastq Line Cnt', nLineCnt)
        return nLineCnt
    else:
        sCmd     = 'wc -l %s > %s'     % (sInFile, sOutFile)
        os.system(sCmd)
        InFile   = open(sOutFile, 'r')
        nLineCnt = int([sReadLine.strip('\n').split(' ') for sReadLine in InFile][0][0])
        InFile.close()
        return nLineCnt
#def END: get_line_cnt


def get_split_list (sWorkDir, sFastqTag):

    sInFile = '%s/%s.split.txt' % (sWorkDir, sFastqTag)
    InFile = open(sInFile, 'r')
    list_sSplits = [sReadLine.strip('\n') for sReadLine in InFile if not sReadLine.startswith('#')]
    InFile.close()
    return list_sSplits
#def END: get_split_list


def mod_sort_by_barcode (sDataDir, sSample, sOutputDir, sBarcodeFile):

    sRE            = '[T]{7}'
    nBarcode3Cut   = 3
    nBarcode5Ext   = 18 #end of barcode - 1
    dict_sBarcodes = load_PE_input(sBarcodeFile)

    print('dict_sBarcodes', len(dict_sBarcodes))
    dict_sOutput   = {}
    sInFile        = '%s/%s' % (sDataDir, sSample)
    InFile         = open(sInFile, 'r')

    nNoMatch       = 0
    for i, sReadLine in enumerate(InFile):

        if i % 4 == 0: sReadID = sReadLine.replace('\n', '')
        if i % 4 != 1: continue

        sNGSSeq = sReadLine.replace('\n', '').upper()

        for sReIndex in regex.finditer(sRE, sNGSSeq, overlapped=True):
            nIndexStart = sReIndex.start()
            nIndexEnd   = sReIndex.end()
            sBarcode    = sNGSSeq[nIndexStart+nBarcode3Cut:nIndexEnd+nBarcode5Ext]

            #if nIndexStart > (len(sNGSSeq) / 2): # Need to improve better way of filtering multi-TTTT reads
            #    continue # SKIP barcode in back of read

            ### Skip Non-barcodes ###
            try:
                cPE = dict_sBarcodes[sBarcode]
                nNoMatch += 1
            except KeyError: continue
            #########################

            if sBarcode not in dict_sOutput:
                dict_sOutput[sBarcode] = []
            dict_sOutput[sBarcode].append([sReadID, sNGSSeq])
        #loop END: i, sReadLine
    #loop END: cPE
    InFile.close()
    print('Barcode Found', len(dict_sOutput))
    print('Barcode Not Found', nNoMatch)

    sOutFile = '%s/%s.output.txt' % (sOutputDir, sSample)
    OutFile  = open(sOutFile, 'w')

    for sBarcode in dict_sOutput:
        for sReadID, sNGSSeq in dict_sOutput[sBarcode]:
            sOut = '%s\t%s\n' % (sBarcode, sNGSSeq)
            OutFile.write(sOut)
        #loop END: sReadID, sNGSSeq
    #loop END: sBarcode
    OutFile.close()
#def END: mod_sort_by_barcode


def endo_sort_by_barcode_step1 (work_dir, dict_data):

    print('Samples', len(dict_data))

    dict_output = {}

    for sample in dict_data:
        if sample != 'B1_Rep1': continue

        amplicon, fastq1, fastq2, barcode1, barcode2 = dict_data[sample]
        b1cnt = 0
        b2cnt = 0
        total = 0
        infastq = '%s/flash/%s/%s.extendedFrags.fastq' % (work_dir, sample, sample)

        for sSeqData in SeqIO.parse(infastq, 'fastq'):
            readID = str(sSeqData.id)
            seq    = str(sSeqData.seq)
            total += 1

            if barcode1 in seq: b1cnt += 1
            if barcode2 in seq: b2cnt += 1

        #loop END: sSeqData

        print('%s,%s,%s,%s' % (sample, b1cnt, b2cnt, total))
    #loop END: sample


    '''
    sOutFile = '%s/%s.output.txt' % (sOutputDir, sSample)
    OutFile  = open(sOutFile, 'w')

    for sBarcode in dict_sOutput:
        for sReadID, sNGSSeq in dict_sOutput[sBarcode]:
            sOut = '%s\t%s\n' % (sBarcode, sNGSSeq)
            OutFile.write(sOut)
        #loop END: sReadID, sNGSSeq
    #loop END: sBarcode
    OutFile.close()'''
#def END: mod_sort_by_barcode


def endo_sort_by_barcode_step2(work_dir, dict_data):

    check             = 0
    include_alignment = 1

    dict_readcnt      = {}
    dict_altcnt       = {}
    list_samples      = list(dict_data.keys())
    #list_samples      = ['CV10_Ctrl']
    #list_samples      = ['B1_Rep2']
    #list_samples      = ['B18_Rep1']

    dict_poscheck     = {}

    for sample in list_samples:

        #if sample.split('_')[0] not in ['B1', 'B3', 'B4', 'B5', 'B9', 'B10']: continue
        if sample not in dict_poscheck: dict_poscheck[sample]  = {'ins': 0, 'dels': 0, 'subs':0}

        if sample not in dict_readcnt: dict_readcnt[sample]   = {'wt':0, 'ed':0, 'other':0, 'nobar':0, 'tot':0}


        barcode  = dict_data[sample][0]
        fastq1   = dict_data[sample][1]
        fastq2   = dict_data[sample][2]
        wtseq    = dict_data[sample][3]
        edseq    = dict_data[sample][4]
        amplicon = dict_data[sample][5]
        nickpos  = dict_data[sample][6]
        rttend   = dict_data[sample][7]
        infastq  = '%s/flash/%s/%s.extendedFrags.fastq' % (work_dir, sample, sample)

        nickwindow   = [i for i in range(nickpos - 5, nickpos + 5)]
        rttwindow    = [i for i in range(rttend - 5, rttend + 5)]
        list_indexes = [i for i in range(nickpos - 5, rttend + 5)]

        for sSeqData in SeqIO.parse(infastq, 'fastq'):
            readID        = str(sSeqData.id)
            seq           = str(sSeqData.seq)
            dict_readcnt[sample]['tot'] += 1

            indexes       = [[reindex.start(), reindex.end()] for reindex in regex.finditer(barcode, seq, overlapped=True)]

            if not indexes:
                dict_readcnt[sample]['nobar'] += 1
                continue  #no barcode

            query         = seq
            ref           = amplicon
            gap_incentive = np.zeros(len(ref) + 1, dtype=int)

            if sample.split('_')[0] in ['B1', 'B3', 'B4', 'B5', 'B9', 'B10']:
                wtseq = reverse_complement(wtseq)
                edseq = reverse_complement(edseq)

            if wtseq in query:   dict_readcnt[sample]['wt'] += 1
            elif edseq in query: dict_readcnt[sample]['ed'] += 1
            else:
                dict_readcnt[sample]['other'] += 1

                if include_alignment:

                    if sample not in dict_altcnt: dict_altcnt[sample] = {'nick': {'ins': 0, 'dels': 0, 'subs': 0},
                                                                         'rtt': {'ins': 0, 'dels': 0, 'subs': 0},
                                                                         'notwin': {'ins': 0, 'dels': 0, 'subs': 0},
                                                                         'total': {'ins': 0, 'dels': 0, 'subs': 0}}



                    align_result = cp2_align.global_align(query,
                                                          ref,
                                                          matrix=ALN_MATRIX,
                                                          gap_open=-10,
                                                          gap_extend=1,
                                                          gap_incentive=gap_incentive)

                    #testseq1 = 'AGGTGTGGATCCAAAGCTTATTTCTAGAATTTGGGTTTATAATCACTATAGATGGATCATATGGAAACTGGCAGCTATGGAATGTGCCTTTCCTAAGGAATTTGCTAATAGATGCCT--G--C---A----T---TCTTCAACTAAAATACAGGCAAGTTTAAAGCATTACATTACGTAATCATATACGGCAGTATGGTTAAGGTTTCTGTGTAGTCTGTGACTTCCATGTCAAAATGTTGCACAAGCCAGTTGTCAGTGACAG'
                    #testseq2 = 'AGGTGTGGATCCAAAGCTTATTTCTAGAATTTGGGTTTATAATCACTATAGATGGATCATATGGAAACTGGCAGCTATGGAATGTGCCTTTCCTAAGGAATTTGCTAATAGATGCCTAAGCCCAGAAAGGGTGCTTCTTCAACTAAAATACAGGCAAGTTTAAAGCATTACATTACGTAATCATATACGGCAGTATGGTTAAGGTTTCTGTGTAGTCTGTGACTTCCATGTCAAAATGTTGCACAAGCCAGTTGTCAGTGACAG'

                    #anno_results = cp2.find_indels_substitutions(testseq1, testseq2, list_indexes)
                    anno_results   = cp2.find_indels_substitutions(align_result[0], align_result[1], list_indexes)
                    list_index_ins = anno_results['insertion_coordinates']
                    list_index_del = anno_results['deletion_coordinates']      #[(117, 119), (120, 122), (123, 126), (127, 131)]
                    list_index_sub = anno_results['all_substitution_positions']

                    dict_poscheck[sample]['ins'] += len(list_index_ins)
                    dict_poscheck[sample]['dels'] += len(list_index_del)
                    dict_poscheck[sample]['subs'] += len(list_index_sub)

                    if check:

                        if list_index_sub:
                            print(barcode)
                            print(align_result[0])
                            print(align_result[1])
                            print(wtseq)
                            print(edseq)
                            print(list_indexes)
                            print(nickwindow)
                            print(rttwindow)


                            for anno in anno_results:
                                if anno.startswith('ref'): continue
                                print(anno, anno_results[anno])


                            print(list_index_sub)

                            print('rtt',    dict_altcnt[sample]['rtt']['subs'])
                            print('notrtt', dict_altcnt[sample]['notwin']['subs'])

                            print('rttidx', [i for i in list_index_sub if i in nickwindow])
                            print('rttnotidx', [i for i in list_index_sub if i not in nickwindow])

                            if dict_altcnt[sample]['nick']['subs'] == 100: sys.exit() #for debugging
                    #if END: check

                    # count sub positions
                    dict_altcnt[sample]['nick']['subs']    += len([i for i in list_index_sub if i in nickwindow])
                    dict_altcnt[sample]['rtt']['subs']     += len([i for i in list_index_sub if i in rttwindow])

                    # count indels coordinates
                    dict_altcnt[sample]['nick']['ins']     += get_coordinate_in_window(list_index_ins, nickwindow)[0]
                    dict_altcnt[sample]['nick']['dels']    += get_coordinate_in_window(list_index_del, nickwindow)[0]
                    dict_altcnt[sample]['rtt']['ins']      += get_coordinate_in_window(list_index_ins, rttwindow)[0]
                    dict_altcnt[sample]['rtt']['dels']     += get_coordinate_in_window(list_index_del, rttwindow)[0]

                    # non window indexes
                    dict_altcnt[sample]['notwin']['subs']  += len([i for i in list_index_sub if i not in nickwindow + rttwindow])
                    dict_altcnt[sample]['notwin']['ins']   += get_coordinate_in_window(list_index_ins, nickwindow + rttwindow)[1]
                    dict_altcnt[sample]['notwin']['dels']  += get_coordinate_in_window(list_index_del, nickwindow + rttwindow)[1]



                #if END:
            #if END:
        # loop END: sSeqData
    # loop END: sample

    for sample in list_samples:
        cnt = '%s,%s' % (','.join([str(dict_poscheck[sample][out]) for out in ['ins', 'dels', 'subs']]),dict_readcnt[sample]['other'])

        if dict_readcnt[sample]['other'] == 0:
            avg = 'no_others'
        else:
            avg = '%s' % (','.join([str( dict_poscheck[sample][out] / dict_readcnt[sample]['other']) for out in ['ins', 'dels', 'subs']]))

        print('%s,%s,%s' % (sample, cnt, avg))

    sys.exit()




    '''
    for sample in list_samples:
        out_nickpos = '%s' % (','.join([str(dict_altcnt[sample]['nick'][out]) for out in ['ins', 'dels', 'subs']]))
        out_notnickpos = '%s' % (','.join([str(dict_altcnt[sample]['notnick'][out]) for out in ['ins', 'dels', 'subs']]))
        
        out_rttpos  = '%s' % (','.join([str(dict_altcnt[sample]['rtt'][out]) for out in ['ins', 'dels', 'subs']]))
        out_notrttpos  = '%s' % (','.join([str(dict_altcnt[sample]['notrtt'][out]) for out in ['ins', 'dels', 'subs']]))
        
        print(out_nickpos, out_rttpos)
        print(out_notnickpos, out_notrttpos)
    sys.exit()

    #for sample in list_samples:
        #if sample.split('_')[0] not in ['B1', 'B3', 'B4', 'B5', 'B9', 'B10']: continue
        #print('%s,%s' % (sample, ','.join([str(dict_output1[sample][out]) for out in ['wt', 'ed', 'other', 'nobar', 'tot']])))
'''

    outfile = '%s/20221116_indelsub_v2.txt' % work_dir
    outf    = open(outfile, 'w')

    for sample in list_samples:
        out_nickpos = '%s' % (','.join([str(dict_altcnt[sample]['nick'][out]) for out in ['ins', 'dels', 'subs']]))
        out_rttpos  = '%s' % (','.join([str(dict_altcnt[sample]['rtt'][out]) for out in ['ins', 'dels', 'subs']]))
        out_notwin  = '%s' % (','.join([str(dict_altcnt[sample]['notwin'][out]) for out in ['ins', 'dels', 'subs']]))

        out = '%s,%s,%s,%s\n' % (sample, out_nickpos, out_rttpos, out_notwin)
        outf.write(out)
    #loop END: sample
    outf.close()
# def END: mod_sort_by_barcode


def endo_sort_by_barcode_readlevelcnt (work_dir, dict_data):

    check             = 0
    include_alignment = 1
    dict_readcnt      = {}
    dict_altcnt       = {}
    list_samples      = list(dict_data.keys())
    #list_samples      = ['CV10_Ctrl']
    #list_samples      = ['B1_Rep2']
    #list_samples      = ['B18_Rep1']

    dict_altpattern = {}
    for sample in list_samples[:1]:

        # if sample.split('_')[0] not in ['B1', 'B3', 'B4', 'B5', 'B9', 'B10']: continue

        if sample not in dict_readcnt: dict_readcnt[sample]   = {'wt': 0, 'ed': 0, 'other': 0, 'nobar': 0, 'tot': 0}

        if sample not in dict_altcnt: dict_altcnt[sample] = {'ins':       0, # insertion only in either nick or rtt window
                                                             'dels':      0, # deletion only ...
                                                             'indels':    0, # both indels
                                                             'subs_only': 0,
                                                             'noalt':     0,} # sub only, if sub is with others, prioritize indels
        barcode     = dict_data[sample][0]
        fastq1      = dict_data[sample][1]
        fastq2      = dict_data[sample][2]
        wtseq       = dict_data[sample][3]
        edseq       = dict_data[sample][4]
        amplicon    = dict_data[sample][5]
        nickpos     = dict_data[sample][6]
        rttend      = dict_data[sample][7]

        infastq     = '%s/flash/%s/%s.extendedFrags.fastq' % (work_dir, sample, sample)

        nickwindow   = [i for i in range(nickpos - 5, nickpos + 5)]
        rttwindow    = [i for i in range(rttend - 5, rttend + 5)]
        list_window  = nickwindow + rttwindow

        for sSeqData in SeqIO.parse(infastq, 'fastq'):
            readID  = str(sSeqData.id)
            seq     = str(sSeqData.seq)
            dict_readcnt[sample]['tot'] += 1

            indexes = [[reindex.start(), reindex.end()] for reindex in
                       regex.finditer(barcode, seq, overlapped=True)]

            if not indexes:
                dict_readcnt[sample]['nobar'] += 1
                continue  # no barcode

            query   = seq
            ref     = amplicon
            gap_incentive = np.zeros(len(ref) + 1, dtype=int)

            if sample.split('_')[0] in ['B1', 'B3', 'B4', 'B5', 'B9', 'B10']:
                wtseq = reverse_complement(wtseq)
                edseq = reverse_complement(edseq)

            if wtseq in query:    dict_readcnt[sample]['wt'] += 1
            elif edseq in query:  dict_readcnt[sample]['ed'] += 1
            else:
                dict_readcnt[sample]['other'] += 1

                if include_alignment:


                    align_result = cp2_align.global_align(query,
                                                          ref,
                                                          matrix=ALN_MATRIX,
                                                          gap_open=-10,
                                                          gap_extend=1,
                                                          gap_incentive=gap_incentive)

                    # testseq1 = 'AGGTGTGGATCCAAAGCTTATTTCTAGAATTTGGGTTTATAATCACTATAGATGGATCATATGGAAACTGGCAGCTATGGAATGTGCCTTTCCTAAGGAATTTGCTAATAGATGCCT--G--C---A----T---TCTTCAACTAAAATACAGGCAAGTTTAAAGCATTACATTACGTAATCATATACGGCAGTATGGTTAAGGTTTCTGTGTAGTCTGTGACTTCCATGTCAAAATGTTGCACAAGCCAGTTGTCAGTGACAG'
                    # testseq2 = 'AGGTGTGGATCCAAAGCTTATTTCTAGAATTTGGGTTTATAATCACTATAGATGGATCATATGGAAACTGGCAGCTATGGAATGTGCCTTTCCTAAGGAATTTGCTAATAGATGCCTAAGCCCAGAAAGGGTGCTTCTTCAACTAAAATACAGGCAAGTTTAAAGCATTACATTACGTAATCATATACGGCAGTATGGTTAAGGTTTCTGTGTAGTCTGTGACTTCCATGTCAAAATGTTGCACAAGCCAGTTGTCAGTGACAG'

                    # anno_results = cp2.find_indels_substitutions(testseq1, testseq2, list_indexes)
                    anno_results   = cp2.find_indels_substitutions(align_result[0], align_result[1], list_window)
                    list_index_ins = anno_results['insertion_coordinates']
                    list_index_del = anno_results['deletion_coordinates']  # [(117, 119), (120, 122), (123, 126), (127, 131)]
                    list_index_sub = anno_results['all_substitution_positions']

                    altpattern = get_alt_pattern (list_index_ins, list_index_del, list_index_sub)

                    if altpattern not in dict_altpattern:
                        dict_altpattern[altpattern] = 0
                    dict_altpattern[altpattern] += 1

                    if altpattern == '001': #only subs
                        dict_altcnt[sample]['subs_only'] += 1

                    elif altpattern in ['100', '101']: #ins only or ins with subs
                        dict_altcnt[sample]['ins'] += 1

                    elif altpattern in ['010', '011']: #dels only or dels with subs
                        dict_altcnt[sample]['dels'] += 1

                    elif altpattern in ['110', '111']: #indels with subs
                        dict_altcnt[sample]['indels'] += 1

                    else: dict_altcnt[sample]['noalt'] += 1

                    if check:
                        print(list_window)
                        print(nickwindow)
                        print(rttwindow)
                        print(altpattern)

                        for anno in anno_results:
                            if anno.startswith('ref'): continue
                            print(anno, anno_results[anno])

                        if dict_readcnt[sample]['other'] == 100: sys.exit()  # for debugging
                    # if END: check
                # if END:
            # if END:
        # loop END: sSeqData
    # loop END: sample

    '''
    for sample in list_samples:
        out_nickpos = '%s' % (','.join([str(dict_altcnt[sample]['nick'][out]) for out in ['ins', 'dels', 'subs']]))
        out_notnickpos = '%s' % (','.join([str(dict_altcnt[sample]['notnick'][out]) for out in ['ins', 'dels', 'subs']]))

        out_rttpos  = '%s' % (','.join([str(dict_altcnt[sample]['rtt'][out]) for out in ['ins', 'dels', 'subs']]))
        out_notrttpos  = '%s' % (','.join([str(dict_altcnt[sample]['notrtt'][out]) for out in ['ins', 'dels', 'subs']]))

        print(out_nickpos, out_rttpos)
        print(out_notnickpos, out_notrttpos)
    sys.exit()

    #for sample in list_samples:
        #if sample.split('_')[0] not in ['B1', 'B3', 'B4', 'B5', 'B9', 'B10']: continue
        #print('%s,%s' % (sample, ','.join([str(dict_output1[sample][out]) for out in ['wt', 'ed', 'other', 'nobar', 'tot']])))
'''

    outfile = '%s/20221116_indelsub_others.txt' % work_dir
    outf = open(outfile, 'w')

    for sample in list_samples:
        list_outkeys = ['ins', 'dels', 'subs_only', 'noalt']
        out = '%s,%s\n' % (sample, ','.join([str(dict_altcnt[sample][out]) for out in list_outkeys]))
        outf.write(out)
    # loop END: sample
    outf.close()
# def END: mod_sort_by_barcode


def endo_sort_by_barcode_readlevelcnt_ctrl (analysis, work_dir, dict_data):
    winbuffer          = 5
    fastqout           = 1
    check              = 0
    include_alignment  = 1
    score_cutoff      = 15

    dict_readcnt       = {}
    dict_altcnt        = {}
    list_sample_tags   = list(dict_data.keys())
    list_samples       = list(set(['_'.join(sample_tag.split('_')[:-1]) for sample_tag in list_sample_tags]))

    #list_samples      = ['CV10_Ctrl']
    #list_samples      = ['B1_Rep2']
    list_samples       = ['PE3-P6_Ctrl']

    dict_altpattern   = {}
    for sample in list_samples:

        print(sample)

        # if sample.split('_')[0] not in ['B1', 'B3', 'B4', 'B5', 'B9', 'B10']: continue

        if sample not in dict_readcnt: dict_readcnt[sample]   = {'wt': 0, 'ed': 0, 'other': 0, 'nobar': 0, 'tot': 0}

        if sample not in dict_altcnt: dict_altcnt[sample] = {'ins':       [], # insertion only in either nick or rtt window
                                                             'dels':      [], # deletion only ...
                                                             'indels':    [], # both indels
                                                             'subs_only': [],
                                                             'noalt':     [],} # sub only, if sub is with others, prioritize indels



        for fastqtag in [7, 8]:

            key = '%s_%s' % (sample, fastqtag)

            barcode     = dict_data[key][0]
            fastq1      = dict_data[key][1]
            fastq2      = dict_data[key][2]
            wtseq       = dict_data[key][3]
            edseq       = dict_data[key][4]
            amplicon    = dict_data[key][5]
            nickpos     = dict_data[key][6]
            rttend      = dict_data[key][7]
            nick_ngrna  = dict_data[key][8]

            infastq     = '%s/flash/%s_%s/%s_%s.extendedFrags.fastq' % (work_dir, sample, fastqtag, sample, fastqtag)

            nickwindow   = [i for i in range(nickpos - winbuffer, nickpos + winbuffer)]
            rttwindow    = [i for i in range(rttend - winbuffer, rttend + winbuffer)]


            if nick_ngrna != 'NA':
                ngrnawindow = [i for i in range(nick_ngrna - winbuffer, nick_ngrna + winbuffer)]
                list_window  = ngrnawindow + nickwindow + rttwindow
            else:
                list_window  = nickwindow + rttwindow

            for seqdata in SeqIO.parse(infastq, 'fastq'):
                readID  = str(seqdata.id)
                seq     = str(seqdata.seq)
                dict_readcnt[sample]['tot'] += 1

                indexes = [[reindex.start(), reindex.end()] for reindex in
                           regex.finditer(barcode, seq, overlapped=True)]

                if not indexes:
                    dict_readcnt[sample]['nobar'] += 1
                    continue  # no barcode

                query         = seq
                ref           = amplicon
                gap_incentive = np.zeros(len(ref) + 1, dtype=int)

                if wtseq in query:    dict_readcnt[sample]['wt'] += 1
                elif edseq in query:  dict_readcnt[sample]['ed'] += 1
                else:
                    dict_readcnt[sample]['other'] += 1

                    if include_alignment:


                        align_result = cp2_align.global_align(query,
                                                              ref,
                                                              matrix=ALN_MATRIX,
                                                              gap_open=-10,
                                                              gap_extend=1,
                                                              gap_incentive=gap_incentive)
                        alignscore = align_result[2]

                        if alignscore <= score_cutoff: continue  #when NGS read length is too small

                        anno_results   = cp2.find_indels_substitutions(align_result[0], align_result[1], list_window)
                        list_index_ins = anno_results['insertion_coordinates']
                        list_index_del = anno_results['deletion_coordinates']  # [(117, 119), (120, 122), (123, 126), (127, 131)]
                        list_index_sub = anno_results['substitution_positions']

                        altpattern = get_alt_pattern (list_index_ins, list_index_del, list_index_sub)

                        if altpattern not in dict_altpattern:
                            dict_altpattern[altpattern] = 0
                        dict_altpattern[altpattern] += 1

                        if altpattern == '001': #only subs
                            dict_altcnt[sample]['subs_only'].append(seqdata)

                        elif altpattern in ['100', '101']: #ins only or ins with subs
                            dict_altcnt[sample]['ins'].append(seqdata)

                        elif altpattern in ['010', '011']: #dels only or dels with subs
                            dict_altcnt[sample]['dels'].append(seqdata)

                        elif altpattern in ['110', '111']: #indels with subs
                            dict_altcnt[sample]['indels'].append(seqdata)

                        else: dict_altcnt[sample]['noalt'].append(seqdata)

                        if check:
                            print(list_window)

                            for anno in anno_results:
                                if anno.startswith('ref'): continue
                                print(anno, anno_results[anno])

                        # if END: check
                    # if END:
                # if END:
            # loop END: sSeqData
        # loop END: fastqtag
    # loop END: sample

    '''
    for sample in list_samples:
        out_nickpos = '%s' % (','.join([str(dict_altcnt[sample]['nick'][out]) for out in ['ins', 'dels', 'subs']]))
        out_notnickpos = '%s' % (','.join([str(dict_altcnt[sample]['notnick'][out]) for out in ['ins', 'dels', 'subs']]))

        out_rttpos  = '%s' % (','.join([str(dict_altcnt[sample]['rtt'][out]) for out in ['ins', 'dels', 'subs']]))
        out_notrttpos  = '%s' % (','.join([str(dict_altcnt[sample]['notrtt'][out]) for out in ['ins', 'dels', 'subs']]))

        print(out_nickpos, out_rttpos)
        print(out_notnickpos, out_notrttpos)
    sys.exit()

    #for sample in list_samples:
        #if sample.split('_')[0] not in ['B1', 'B3', 'B4', 'B5', 'B9', 'B10']: continue
        #print('%s,%s' % (sample, ','.join([str(dict_output1[sample][out]) for out in ['wt', 'ed', 'other', 'nobar', 'tot']])))
'''


    outfile  = '%s/%s_readcnt_wt_ed_update%s.txt' % (work_dir, analysis, '_check' if check else '')
    outfile2 = '%s/%s_others_indel_update%s.txt'  % (work_dir, analysis, '_check' if check else '')
    outf1    = open(outfile, 'w')
    outf2    = open(outfile2, 'w')

    list_outkeys1 = ['wt', 'ed', 'other', 'nobar', 'tot']
    list_outkeys2 = ['ins', 'dels', 'subs_only', 'noalt']

    outf1.write('sample,%s\n' % ','.join(list_outkeys1))
    outf2.write('sample,%s\n' % ','.join(list_outkeys2))

    for sample in list_samples:

        out          = '%s,%s\n' % (sample, ','.join([str(dict_readcnt[sample][out]) for out in list_outkeys1]))
        outf1.write(out)

        out          = '%s,%s\n' % (sample, ','.join([str(len(dict_altcnt[sample][out])) for out in list_outkeys2]))
        outf2.write(out)


        if fastqout:

            for out in list_outkeys2:
                outfastq = open('%s/%s-%s.fastq' % (work_dir, sample, out), 'w')

                for seqdata in dict_altcnt[sample][out]:

                    SeqIO.write(seqdata, outfastq, 'fastq')
                #loop END: seqdata
                outfastq.close()
            #loop END: out


    # loop END: sample
    outf1.close()
    outf2.close()

# def END: mod_sort_by_barcode


def endo_sort_by_barcode_readlevelcnt_pe3 (analysis, work_dir, dict_data):

    winbuffer         = 5
    check             = 0
    fastqout          = 1
    include_alignment = 1
    score_cutoff      = 60


    dict_readcnt      = {}
    dict_altcnt       = {}
    list_samples      = list(dict_data.keys())
    #list_samples      = ['PE3b-P10_PE3b-n8_Rep1']

    dict_altpattern = {}

    for sample in list_samples:

        print(sample)

        # if sample.split('_')[0] not in ['B1', 'B3', 'B4', 'B5', 'B9', 'B10']: continue

        if sample not in dict_readcnt: dict_readcnt[sample]   = {'wt': 0, 'ed': 0, 'other': 0, 'nobar': 0, 'tot': 0}

        if sample not in dict_altcnt: dict_altcnt[sample]   = {'ins':       [], # insertion only in either nick or rtt window
                                                               'dels':      [], # deletion only ...
                                                               'indels':    [], # both indels
                                                               'subs_only': [],
                                                               'noalt':     [],} # sub only, if sub is with others, prioritize indels

        infastq     = '%s/flash/%s/%s.extendedFrags.fastq' % (work_dir, sample, sample)
        barcode     = dict_data[sample][0]
        fastq1      = dict_data[sample][1]
        fastq2      = dict_data[sample][2]
        wtseq       = dict_data[sample][3]
        edseq       = dict_data[sample][4]
        amplicon    = dict_data[sample][5]
        nickpos     = dict_data[sample][6]
        rttend      = dict_data[sample][7]

        ## Set Target Indexes ##
        nickwindow = [i for i in range(nickpos - winbuffer, nickpos + winbuffer)]
        rttwindow  = [i for i in range(rttend - winbuffer, rttend + winbuffer)]

        if analysis == 'ENDO_221119_PE2_ADD':
            revcom       = dict_data[sample][8]
            list_window  = nickwindow + rttwindow
            if revcom:
                wtseq = reverse_complement(wtseq)
                edseq = reverse_complement(edseq)

        else:
            nick_ngrna   = dict_data[sample][8]

            if nick_ngrna != 'NA':
                ngrnawindow = [i for i in range(nick_ngrna - winbuffer, nick_ngrna + winbuffer)]
                list_window  = ngrnawindow + nickwindow + rttwindow
            else:
                list_window  = nickwindow + rttwindow

        for seqdata in SeqIO.parse(infastq, 'fastq'):
            readID  = str(seqdata.id)
            seq     = str(seqdata.seq)
            dict_readcnt[sample]['tot'] += 1

            indexes = [[reindex.start(), reindex.end()] for reindex in
                       regex.finditer(barcode, seq, overlapped=True)]


            if not indexes:
                dict_readcnt[sample]['nobar'] += 1
                continue  # no barcode

            query   = seq
            ref     = amplicon
            gap_incentive = np.zeros(len(ref) + 1, dtype=int)


            if wtseq in query:    dict_readcnt[sample]['wt'] += 1
            elif edseq in query:  dict_readcnt[sample]['ed'] += 1
            else:
                dict_readcnt[sample]['other'] += 1

                if include_alignment:


                    align_result = cp2_align.global_align(query,
                                                          ref,
                                                          matrix=ALN_MATRIX,
                                                          gap_open=-10,
                                                          gap_extend=1,
                                                          gap_incentive=gap_incentive)
                    alignscore   = align_result[2]
                    if alignscore <= score_cutoff: continue  # when NGS read length is too small

                    # testseq1 = 'AGGTGTGGATCCAAAGCTTATTTCTAGAATTTGGGTTTATAATCACTATAGATGGATCATATGGAAACTGGCAGCTATGGAATGTGCCTTTCCTAAGGAATTTGCTAATAGATGCCT--G--C---A----T---TCTTCAACTAAAATACAGGCAAGTTTAAAGCATTACATTACGTAATCATATACGGCAGTATGGTTAAGGTTTCTGTGTAGTCTGTGACTTCCATGTCAAAATGTTGCACAAGCCAGTTGTCAGTGACAG'
                    # testseq2 = 'AGGTGTGGATCCAAAGCTTATTTCTAGAATTTGGGTTTATAATCACTATAGATGGATCATATGGAAACTGGCAGCTATGGAATGTGCCTTTCCTAAGGAATTTGCTAATAGATGCCTAAGCCCAGAAAGGGTGCTTCTTCAACTAAAATACAGGCAAGTTTAAAGCATTACATTACGTAATCATATACGGCAGTATGGTTAAGGTTTCTGTGTAGTCTGTGACTTCCATGTCAAAATGTTGCACAAGCCAGTTGTCAGTGACAG'

                    # anno_results = cp2.find_indels_substitutions(testseq1, testseq2, list_indexes)
                    anno_results   = cp2.find_indels_substitutions(align_result[0], align_result[1], list_window)
                    list_index_ins = anno_results['insertion_coordinates']
                    list_index_del = anno_results['deletion_coordinates']  # [(117, 119), (120, 122), (123, 126), (127, 131)]
                    list_index_sub = anno_results['substitution_positions']

                    altpattern = get_alt_pattern (list_index_ins, list_index_del, list_index_sub)

                    if altpattern not in dict_altpattern:
                        dict_altpattern[altpattern] = 0
                    dict_altpattern[altpattern] += 1

                    if altpattern == '001': #only subs
                        dict_altcnt[sample]['subs_only'].append(seqdata)

                    elif altpattern in ['100', '101']: #ins only or ins with subs
                        dict_altcnt[sample]['ins'].append(seqdata)

                    elif altpattern in ['010', '011']: #dels only or dels with subs
                        dict_altcnt[sample]['dels'].append(seqdata)

                    elif altpattern in ['110', '111']: #indels with subs
                        dict_altcnt[sample]['indels'].append(seqdata)

                    else: dict_altcnt[sample]['noalt'].append(seqdata)

                    if check:
                        print(list_window)
                        print(altpattern)

                        for anno in anno_results:
                            if anno.startswith('ref'): continue
                            print(anno, anno_results[anno])

                    # if END: check
                # if END:
            # if END:
        # loop END: sSeqData
    # loop END: sample
    if check: sys.exit()

    '''
    for sample in list_samples:
        out_nickpos = '%s' % (','.join([str(dict_altcnt[sample]['nick'][out]) for out in ['ins', 'dels', 'subs']]))
        out_notnickpos = '%s' % (','.join([str(dict_altcnt[sample]['notnick'][out]) for out in ['ins', 'dels', 'subs']]))

        out_rttpos  = '%s' % (','.join([str(dict_altcnt[sample]['rtt'][out]) for out in ['ins', 'dels', 'subs']]))
        out_notrttpos  = '%s' % (','.join([str(dict_altcnt[sample]['notrtt'][out]) for out in ['ins', 'dels', 'subs']]))

        print(out_nickpos, out_rttpos)
        print(out_notnickpos, out_notrttpos)
    sys.exit()

    #for sample in list_samples:
        #if sample.split('_')[0] not in ['B1', 'B3', 'B4', 'B5', 'B9', 'B10']: continue
        #print('%s,%s' % (sample, ','.join([str(dict_output1[sample][out]) for out in ['wt', 'ed', 'other', 'nobar', 'tot']])))
'''


    outfile  = '%s/%s_readcnt_wt_ed_update.txt' % (work_dir, analysis)
    outfile2 = '%s/%s_others_indel_update.txt'  % (work_dir, analysis)
    outf1    = open(outfile, 'w')
    outf2    = open(outfile2, 'w')

    list_outkeys1 = ['wt', 'ed', 'other', 'nobar', 'tot']
    list_outkeys2 = ['ins', 'dels', 'subs_only', 'noalt']

    outf1.write('%s\n' % ','.join(list_outkeys1))
    outf2.write('%s\n' % ','.join(list_outkeys2))

    for sample in list_samples:

        out          = '%s,%s\n' % (sample, ','.join([str(dict_readcnt[sample][out]) for out in list_outkeys1]))
        outf1.write(out)

        out          = '%s,%s\n' % (sample, ','.join([str(len(dict_altcnt[sample][out])) for out in list_outkeys2]))
        outf2.write(out)


        if fastqout:

            for out in list_outkeys2:
                outfastq = open('%s/%s-%s.fastq' % (work_dir, sample, out), 'w')

                for seqdata in dict_altcnt[sample][out]:

                    SeqIO.write(seqdata, outfastq, 'fastq')
                #loop END: seqdata
                outfastq.close()
            #loop END: out

        #if END: fastqout


    # loop END: sample
    outf1.close()
    outf2.close()
# def END: mod_sort_by_barcode


def endo_sort_by_barcode_readlevelcnt_pe3_OFF (analysis, work_dir, outputdir, dict_data):

    check             = 0
    fastqout1         = 0
    fastqout2         = 1
    newbar            = 0
    use_altseqs       = 1
    score_cutoff      = 75


    dict_altcnt_off   = {}
    dict_altcnt       = {}
    list_groups      = list(dict_data.keys())
    #list_groups      = ['O1-1']

    dict_fastqout2    = {}

    for group in list_groups:

        dict_altpattern = {'off': {}, 'other': {}}

        list_samples = dict_data[group]

        for sample in list_samples:

            print(sample)

            # if sample.split('_')[0] not in ['B1', 'B3', 'B4', 'B5', 'B9', 'B10']: continue
            if sample not in dict_fastqout2: dict_fastqout2[sample] = []
            if sample not in dict_altcnt_off: dict_altcnt_off[sample]    = {'ins':       [], # insertion only in either nick or rtt window
                                                                            'dels':      [], # deletion only ...
                                                                            'indels':    [], # both indels
                                                                            'subs_only': [],
                                                                            'noalt':     []} # sub only, if sub is with others, prioritize indels

            if sample not in dict_altcnt: dict_altcnt[sample]    = {'ins':       [], # insertion only in either nick or rtt window
                                                                    'dels':      [], # deletion only ...
                                                                    'indels':    [], # both indels
                                                                    'subs_only': [],
                                                                    'noalt':     []} # sub only, if sub is with others, prioritize indels

            infastq     = '%s/flash/%s/%s.extendedFrags.fastq' % (work_dir, sample, sample)


            barcode     = dict_data[group][sample][0]
            wtseq       = dict_data[group][sample][3]
            amplicon    = dict_data[group][sample][4]
            alt_wtseq1  = dict_data[group][sample][5]# to use after initial analysis if needed
            #barcode_ori = dict_data[group][sample][6]


            for seqdata in SeqIO.parse(infastq, 'fastq'):
                readID  = str(seqdata.id)
                seq     = str(seqdata.seq)

                target_barcode = barcode

                bar_indexes    = [[reindex.start(), reindex.end()] for reindex in
                                   regex.finditer(target_barcode, seq, overlapped=True)]
                if not bar_indexes:
                    continue  # no barcode

                dict_fastqout2[sample].append(seqdata)

                wtseq_start, wtseq_end = [[reindex.start(), reindex.end()] for reindex in
                                           regex.finditer(wtseq, amplicon, overlapped=True)][0]

                list_window  = [i for i in range(wtseq_start, wtseq_end)]
                query        = seq
                ref          = amplicon
                alt_ref      = amplicon[:wtseq_start] + alt_wtseq1 + amplicon[wtseq_end:]
                gap_incent   = np.zeros(len(ref) + 1, dtype=int)

                offpattern   = get_off_altpattern (seqdata, alt_ref, ref, list_window)

                align_result = cp2_align.global_align(query,
                                                      ref,
                                                      matrix=ALN_MATRIX,
                                                      gap_open=-10,
                                                      gap_extend=1,
                                                      gap_incentive=gap_incent)
                alignscore     = align_result[2]
                if alignscore <= score_cutoff: continue  # when NGS read length is too small

                anno_results   = cp2.find_indels_substitutions(align_result[0], align_result[1], list_window)
                list_index_ins = anno_results['insertion_coordinates']
                list_index_del = anno_results['deletion_coordinates']  # [(117, 119), (120, 122), (123, 126), (127, 131)]
                list_index_sub = anno_results['substitution_positions']
                list_nt_sub    = anno_results['substitution_values']
                ngspattern     = ['%s-%s' % (i, nt) for i, nt in zip(list_index_sub, list_nt_sub)]

                patterncheck    = [pattern for pattern in ngspattern if pattern in offpattern]
                #targetseq      = ''.join([align_result[0][i].lower() if i in list_index_sub else align_result[0][i] for i in list_window])

                if patterncheck:

                    list_after17 = [i for i in list_index_sub if i >= wtseq_start+17]

                    if list_after17:

                        altpattern = get_alt_pattern (list_index_ins, list_index_del, list_after17)

                        if altpattern == '001': #only subs
                            dict_altcnt_off[sample]['subs_only'].append(seqdata)

                            alt_indexes = [int(pat.split('-')[0]) for pat in patterncheck if int(pat.split('-')[0]) in list_after17]

                            altseq = ''.join([align_result[0][i].lower() if i in alt_indexes else align_result[0][i] for i in list_window])
                            key    = '%s,%s' % (':'.join(patterncheck), altseq)

                            if key not in dict_altpattern['off']:
                                dict_altpattern['off'][key] = {sample:0 for sample in list_samples}

                            dict_altpattern['off'][key][sample] += 1


                        elif altpattern in ['100', '101']: #ins only or ins with subs
                            dict_altcnt_off[sample]['ins'].append(seqdata)

                        elif altpattern in ['010', '011']: #dels only or dels with subs
                            dict_altcnt_off[sample]['dels'].append(seqdata)

                        elif altpattern in ['110', '111']: #indels with subs
                            dict_altcnt_off[sample]['indels'].append(seqdata)

                        else: dict_altcnt_off[sample]['noalt'].append(seqdata)

                else:

                    altpattern = get_alt_pattern (list_index_ins, list_index_del, list_index_sub)

                    if altpattern == '001': #only subs
                        dict_altcnt[sample]['subs_only'].append(seqdata)
                        altseq = ''.join([align_result[0][i].lower() if i in list_index_sub else align_result[0][i] for i in list_window])
                        key = '%s,%s' % (':'.join(ngspattern), altseq)

                        if key not in dict_altpattern['other']:
                            dict_altpattern['other'][key] = {sample:0 for sample in list_samples}

                        dict_altpattern['other'][key][sample] += 1


                    elif altpattern in ['100', '101']: #ins only or ins with subs
                        dict_altcnt[sample]['ins'].append(seqdata)

                    elif altpattern in ['010', '011']: #dels only or dels with subs
                        dict_altcnt[sample]['dels'].append(seqdata)

                    elif altpattern in ['110', '111']: #indels with subs
                        dict_altcnt[sample]['indels'].append(seqdata)

                    else: dict_altcnt[sample]['noalt'].append(seqdata)
                #if END:

                if check:
                    if altpattern in ['001']:

                        print(readID)
                        print(barcode)
                        print(amplicon)
                        print(seq)
                        print(wtseq)
                        print(list_window)
                        print(align_result[0])
                        print(''.join([green(align_result[0][i]) if i in list_index_sub else align_result[0][i] for i in range(len(align_result[0]))]))
                        print(''.join([red(align_result[1][i]) if i in list_index_sub else align_result[1][i] for i in range(len(align_result[1]))]))
                        print(align_result[2])

                        for anno in anno_results:
                            if anno.startswith('ref'): continue
                            print(anno, anno_results[anno])
                        print('COUNT---------------------------> ', len(dict_altcnt[sample]['subs_only']))
                # if END: check
            # loop END: seqdata
        # loop END: sample
        outdir            = '%s/%s/output_patterncheck' % (outputdir, analysis)
        os.makedirs(outdir, exist_ok=True)

        outfile       = '%s/%s_pattercheck.csv' % (outdir, group)
        outf          = open(outfile, 'w')

        for cat in dict_altpattern:
            for pattern in dict_altpattern[cat]:

                list_samples = dict_altpattern[cat][pattern]
                out          = ','.join(['%s,%s' % (sample, dict_altpattern[cat][pattern][sample]) for sample in list_samples])
                outf.write('%s,%s,%s\n' % (cat,pattern, out))
            #loop END: pattern
        #loop END: cat
        outf.close()
    # loop END: group




    '''


    
    
    for sample in list_samples:
        out_nickpos = '%s' % (','.join([str(dict_altcnt[sample]['nick'][out]) for out in ['ins', 'dels', 'subs']]))
        out_notnickpos = '%s' % (','.join([str(dict_altcnt[sample]['notnick'][out]) for out in ['ins', 'dels', 'subs']]))

        out_rttpos  = '%s' % (','.join([str(dict_altcnt[sample]['rtt'][out]) for out in ['ins', 'dels', 'subs']]))
        out_notrttpos  = '%s' % (','.join([str(dict_altcnt[sample]['notrtt'][out]) for out in ['ins', 'dels', 'subs']]))

        print(out_nickpos, out_rttpos)
        print(out_notnickpos, out_notrttpos)
    sys.exit()

    #for sample in list_samples:
        #if sample.split('_')[0] not in ['B1', 'B3', 'B4', 'B5', 'B9', 'B10']: continue
        #print('%s,%s' % (sample, ','.join([str(dict_output1[sample][out]) for out in ['wt', 'ed', 'other', 'nobar', 'tot']])))
'''




    outfile       = '%s/%s_sindelpattern_%s%s%s%s.txt' % (work_dir, analysis, score_cutoff, '_newbar' if newbar else '', '_secondWT' if use_altseqs else '', '_check' if check else '')
    outf1         = open(outfile, 'w')

    list_outkeys  = ['wt', 'ed', 'other', 'nobar', 'tot']
    list_outkeys1 = ['ins', 'dels', 'subs_only', 'noalt']

    outf1.write('sample,%s,%s\n' % (','.join(list_outkeys1), ','.join(['off-%s' % key for key in list_outkeys1])))

    for group in list_groups:

        list_samples = dict_data[group]

        for sample in list_samples:

            out          = '%s,%s,%s\n' % (sample, ','.join([str(len(dict_altcnt[sample][out])) for out in list_outkeys1]), ','.join([str(len(dict_altcnt_off[sample][out])) for out in list_outkeys1]))
            outf1.write(out)

            if fastqout1:

                for out in list_outkeys1:
                    outfastq  = open('%s/%s-%s_others.fastq' % (work_dir, sample, out), 'w')

                    for seqdata in dict_altcnt[sample][out]:
                        SeqIO.write(seqdata, outfadstq, 'fastq')
                    #loop END: seqdata
                    outfastq.close()

                    outfastq2 = open('%s/%s-%s_off.fastq' % (work_dir, sample, out), 'w')
                    for seqdata in dict_altcnt_off[sample][out]:
                        SeqIO.write(seqdata, outfastq2, 'fastq')
                    #loop END: seqdata
                    outfastq2.close()
                #loop END: out
            #if END: fastqout1
        # loop END: sample
    outf1.close()
    # loop END: group
    sys.exit()
    if fastqout2: #for testing mutect, get fastq with just barcoded reads

        for group in list_groups:
            list_samples = dict_data[group]
            for samples in list_samples:

                sample, type = samples.split('_')

                outdir    = '%s/formutect/%s' % (work_dir, sample)
                os.makedirs(outdir, exist_ok=True)

                outfastq  = open('%s/%s.fastq' % (outdir, samples), 'w')

                for seqdata in dict_fastqout2[samples]:
                    SeqIO.write(seqdata, outfastq, 'fastq')
                #loop END: seqdata
                outfastq.close()

            #loop END: sample
        #loop END: group
    #if END: fastqout2
# def END: mod_sort_by_barcode


def get_off_altpattern (seqdata, query, ref, list_window):

    gap_incentive = np.zeros(len(ref) + 1, dtype=int)

    align_result  = cp2_align.global_align(query,
                                           ref,
                                           matrix=ALN_MATRIX,
                                           gap_open=-10,
                                           gap_extend=1,
                                           gap_incentive=gap_incentive)


    anno_results = cp2.find_indels_substitutions(align_result[0], align_result[1], list_window)

    list_index_sub = anno_results['substitution_positions']
    list_nt_sub    = anno_results['substitution_values']

    offpattern     = ['%s-%s' % (i, nt) for i, nt in zip(list_index_sub, list_nt_sub)]

    return offpattern
#def END: get_altpattern


def run_maund (work_dir, dict_data):

    maund             = '%s/maund/maund.py' % sSRC_DIR
    winbuffer         = 5
    check             = 0
    include_alignment = 1
    dict_readcnt      = {}
    dict_altcnt       = {}
    list_samples      = list(dict_data.keys())

    for sample in list_samples[:1]:

        if sample not in dict_readcnt: dict_readcnt[sample]   = {'wt': 0, 'ed': 0, 'other': 0, 'nobar': 0, 'tot': 0}

        if sample not in dict_altcnt: dict_altcnt[sample]   = {'ins':       0, # insertion only in either nick or rtt window
                                                               'dels':      0, # deletion only ...
                                                               'indels':    0, # both indels
                                                               'subs_only': 0,
                                                               'noalt':     0,} # sub only, if sub is with others, prioritize indels
        barcode     = dict_data[sample][0]
        fastq1      = dict_data[sample][1]
        fastq2      = dict_data[sample][2]
        wtseq       = dict_data[sample][3]
        edseq       = dict_data[sample][4]
        amplicon    = dict_data[sample][5]
        nickpos     = dict_data[sample][6]
        rttend      = dict_data[sample][7]
        nick_ngrna  = dict_data[sample][7]
        infastq     = '%s/flash/%s/%s.extendedFrags.fastq' % (work_dir, sample, sample)

        script      = '%s %s %s %s --reverse_complement_match %s' % (maund, amplicon, 'GTGATGAAGGAGATGGGAGG', infastq, reverse_complement('GTGATGAAGGAGATGGGAGG'))

        print(script)
        #os.system(script)
        '''
        ## Set Target Indexes ##
        ngrnawindow  = [i for i in range(nick_ngrna - winbuffer, nick_ngrna + winbuffer)]
        nickwindow   = [i for i in range(nickpos - winbuffer, nickpos + winbuffer)]
        rttwindow    = [i for i in range(rttend - winbuffer, rttend + winbuffer)]

        list_window  = ngrnawindow + nickwindow + rttwindow

        for sSeqData in SeqIO.parse(infastq, 'fastq'):
            readID  = str(sSeqData.id)
            seq     = str(sSeqData.seq)

            indexes = [[reindex.start(), reindex.end()] for reindex in
                       regex.finditer(barcode, seq, overlapped=True)]

            if not indexes:
                dict_readcnt[sample]['nobar'] += 1
                continue  # no barcode
            query   = seq
            ref     = amplicon
        '''



    pass
#def END: run_maund

def get_alt_pattern (list1, list2, list3):

    bool1, bool2, bool3 = 0, 0, 0

    if list1: bool1 = 1
    if list2: bool2 = 1
    if list3: bool3 = 1

    key = '%s%s%s' % (bool1, bool2, bool3)

    return key
#def END: get_alt_pattern



def get_coordinate_in_window (coords, window):
    first    = window[0]
    last     = window[-1]
    list_in  = [[s,e] for s,e in coords if overlap([s,e], [first,last]) > 0]
    list_out = [[s,e] for s,e in coords if overlap([s,e], [first,last]) == 0]
    return [len(list_in), len(list_out)]
#def END: get_coordinate_in_window


def overlap(region1, region2):
    # coordinate is closed at both ends
    start1, end1 = region1
    start2, end2 = region2
    assert (start1 < end1) and (start2 < end2)
    max_start = max([start1, start2])
    min_end   = min([end1,   end2  ])
    length = min_end - max_start + 1
    return length
#def END: overlap


def mp_sort_by_barcode (nCores, sRun, sSample, sInDir, sOutputDir, sBarcodeFile, list_sSplits, sRE, sError, sTopBarcodeFile, nBins):

    sOutDir = '%s/%s' % (sOutputDir, sError)
    os.makedirs(sOutDir, exist_ok=True)

    #dict_cPE            = load_PE_input_v2(sBarcodeFile)

    list_sParameters = []
    for sSplitFile in list_sSplits:
        # HKK_191230_1.extendedFrags_|fastq_01|.fq
        sSplitTag     = '_'.join(sSplitFile.split('.')[1].split('_')[-2:])

        sInFile       = '%s/split/%s'   % (sInDir, sSplitFile)
        sTempDir      = '%s/temp/%s/%s' % (sOutDir, sSample, sSplitTag)
        os.makedirs(sTempDir, exist_ok=True)

        list_sParameters.append([sSplitTag, sInFile, sTempDir, sBarcodeFile, sRE, sError, sRun, nBins, sOutDir])
    #loop END: sSplitFile
    #sort_by_barcode_vOfftarget3(list_sParameters[0])
    #determine_output_vOfftarget3(list_sParameters[0])

    if sRun == '':
        p = mp.Pool(nCores)
        p.map_async(sort_by_barcode_NULL, list_sParameters).get()
        p.map_async(determine_output_NULL, list_sParameters).get()

    elif sRun in ['Ori']:
        ## Original ##
        p = mp.Pool(nCores)
        p.map_async(sort_by_barcode, list_sParameters).get()
        p.map_async(determine_output, list_sParameters).get()

    elif sRun in ['Offtarget3']:
        p = mp.Pool(nCores)
        p.map_async(sort_by_barcode_vOfftarget3, list_sParameters).get()
        p.map_async(determine_output_vOfftarget3, list_sParameters).get()

    else:
        ## V2 - 600K extended barcode version ##
        p = mp.Pool(nCores)
        p.map_async(sort_by_barcode_v3, list_sParameters).get()

        if sRun == 'Offtarget2':
            p.map_async(determine_output_vOfftarget2, list_sParameters).get()

        if sRun in ['Offtarget2-Intended', 'Offtarget2-Intended-Test']: #With extra columns for intended and mismatch edits
            p.map_async(determine_output_vOfftarget2_Intended, list_sParameters).get()

        else:
            pass
            p.map_async(determine_output_v2, list_sParameters).get()
    #if END:
    #########################################
#def END: mp_sort_by_barcode


def mp_sort_by_barcode_vJustNGS (nCores, sRun, sSample, sInDir, sOutDir, sBarcodeFile, list_sSplits, sRE, sError):

    sTemp_RunList  = '%s/run_list2.txt' % sOutDir
    list_sRunFiles = ['%s.extendedFrags_%s.fq' % (sSample, sReadLine.strip('\n')) for sReadLine in open(sTemp_RunList)]

    list_sNewList  = list(set(list_sSplits) - set(list_sRunFiles))

    nBins     = 2
    nTotalCnt = len(list_sNewList)
    list_nBin = [[int(nTotalCnt * (i + 0) / nBins), int(nTotalCnt * (i + 1) / nBins)] for i in range(nBins)]

    sRun      = 0
    list_sNewSplits = list_sNewList[list_nBin[sRun][0]:list_nBin[sRun][1]]

    nTotal     = len(list_sNewSplits)
    list_sParameters  = []
    for i, sSplitFile in enumerate(list_sNewSplits):

        #print(sSplitFile)
        # HKK_191230_1.extendedFrags_|fastq_01|.fq
        sSplitTag = '_'.join(sSplitFile.split('.')[1].split('_')[-2:])
        sInFile   = '%s/split/%s' % (sInDir, sSplitFile)
        assert os.path.isfile(sInFile)

        sTempDir  = '%s/%s/temp/%s/%s' % (sOutDir, sError, sSample, sSplitTag)
        sNGSOut   = '%s/JustNGS/%s'    % (sOutDir, sSplitTag)
        os.makedirs(sTempDir, exist_ok=True)
        os.makedirs(sNGSOut, exist_ok=True)

        nCntTag    = '%s/%s' % (i+1, nTotal)

        list_sParameters.append([sSplitTag, sTempDir, sNGSOut, sBarcodeFile, nCntTag])
    #loop END: sSplitFile

    #determine_output_vJustNGS(list_sParameters[0])
    p = mp.Pool(nCores)
    p.map_async(determine_output_vJustNGS, list_sParameters).get()
#def END: mp_sort_by_barcode_vJustNGS


def sort_by_barcode_NULL (list_sParameters):

    print('Processing NULL %s' % list_sParameters[1])

    sSplitTag      = list_sParameters[0]
    sInFastq       = list_sParameters[1]
    sTempOut       = list_sParameters[2]
    sBarcodeFile   = list_sParameters[3]

    dict_sBarcodes = load_PE_input_v2(sBarcodeFile)

    dict_sOutput   = {}
    InFile         = open(sInFastq, 'r')
    for i, sReadLine in enumerate(InFile):

        if i % 4 == 0: sReadID = sReadLine.replace('\n', '')
        if i % 4 != 1: continue

        sNGSSeq = sReadLine.replace('\n', '').upper()

        for sBarcode in dict_sBarcodes:

            sRE = sBarcode

            for sReIndex in regex.finditer(sRE, sNGSSeq, overlapped=True):
                nIndexStart  = sReIndex.start()
                nIndexEnd    = sReIndex.end()
                #sBarcode   = sNGSSeq[nIndexStart+nBarcode3Cut:nIndexEnd+nBarcode5Ext] if RE = [T]{4}
                sBarcode     = sNGSSeq[nIndexStart:nIndexEnd]
                sRefSeqCheck = sNGSSeq[:nIndexStart]

                if nIndexStart > (len(sNGSSeq) / 2): continue # SKIP barcode in back of read

                ### Skip Non-barcodes ###
                try: cPE = dict_sBarcodes[sBarcode]
                except KeyError:continue
                #########################

                if sBarcode not in dict_sOutput:
                    dict_sOutput[sBarcode] = []
                dict_sOutput[sBarcode].append([sReadID, sNGSSeq])
            #loop END: sReIndex
        #loop END: cPE
    #loop END: i, sReadLine
    InFile.close()

    ## Pickle Out ##
    sOutFile = '%s/%s.data' % (sTempOut, sSplitTag)
    OutFile = open(sOutFile, 'wb')
    pickle.dump(dict_sOutput, OutFile)
    OutFile.close()
#def END: sort_by_barcode_NULL


def determine_output_NULL (list_sParameters):

    print('Processing NULL %s' % list_sParameters[1])
    sSplitTag      = list_sParameters[0]
    sInFastq       = list_sParameters[1]
    sTempOut       = list_sParameters[2]
    sBarcodeFile   = list_sParameters[3]
    dict_cPE       = load_PE_input_v2(sBarcodeFile)
    nBarcodeBuffer = 50 #end of barcode to target seq

    ## Pickle Load ##
    sInFile        = '%s/%s.data' % (sTempOut, sSplitTag)
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

        for sReadID, sNGSSeq in dict_sBarcodes[sBarcode]:

            nBarcodeS      = sNGSSeq.find(sBarcode)
            nBarcodeE      = nBarcodeS + len(sBarcode)
            sWTSeqCheck    = sNGSSeq[nBarcodeE+nBarcodeBuffer:nBarcodeE+nBarcodeBuffer+nWTSize]
            sAltSeqCheck   = sNGSSeq[nBarcodeE+nBarcodeBuffer:nBarcodeE+nBarcodeBuffer+nAltSize]

            if sWTSeqCheck == cPE.sWTSeq:
                dict_sOutput[cPE.sBarcode]['WT'].append(sReadID)

            elif sAltSeqCheck == cPE.sAltSeq:
                dict_sOutput[cPE.sBarcode]['Alt'].append(sReadID)

            elif sWTSeqCheck != cPE.sWTSeq and sAltSeqCheck != cPE.sAltSeq:
                dict_sOutput[cPE.sBarcode]['Other'].append(sReadID)
            #if END:

        #loop END: sReadID, sNGSSeq
    #loop END: sBarcode
    list_sKeys = ['WT', 'Alt', 'Other']

    sOutFile   = '%s/%s.reads.txt' % (sTempOut, sSplitTag)
    OutFile    = open(sOutFile, 'w')

    for sBarcode in dict_sOutput:
        sOut = '%s\t%s\n' % (sBarcode, '\t'.join([','.join(dict_sOutput[sBarcode][sType]) for sType in list_sKeys]))
        OutFile.write(sOut)
    # loop END: sBarcode
    OutFile.close()
#def END: determine_output_NULL


def sort_by_barcode (list_sParameters):

    print('Processing %s' % list_sParameters[1])

    sSplitTag      = list_sParameters[0]
    sInFastq       = list_sParameters[1]
    sTempOut       = list_sParameters[2]
    sBarcodeFile   = list_sParameters[3]
    sRE            = list_sParameters[4]
    sError         = list_sParameters[5]

    nBarcode3Cut   = 3  #
    #nBarcode5Ext   = 15 #end of barcode - 1 # For HKK data      # No need for Single Barcode per Target
    #nBarcode5Ext   = 18 #end of barcode - 1 # For Gyoosang data # No need for Single Barcode per Target
    dict_sBarcodes = load_PE_input(sBarcodeFile)


    print('Bio.SeqIO Version - Sort by Barcode Running - %s' % (list_sParameters[0]))

    dict_sOutput   = {}
    dict_sOutput2  = {}
    InFile         = open(sInFastq, 'r')
    for sSeqData in SeqIO.parse(InFile, 'fastq'):

        sReadID = str(sSeqData.id)
        sNGSSeq = str(sSeqData.seq)

        for sReIndex in regex.finditer(sRE, sNGSSeq, overlapped=True):
            nIndexStart   = sReIndex.start()
            nIndexEnd     = sReIndex.end()
            sBarcodeMatch = sNGSSeq[nIndexStart+nBarcode3Cut:nIndexEnd] #if RE = [T]{4}
            #sBarcode     = sNGSSeq[nIndexStart:nIndexEnd]
            sRefSeqCheck  = sNGSSeq[:nIndexStart]

            #if nIndexStart > (len(sNGSSeq) / 2): continue # SKIP barcode in back of read
            #print(sBarcode)
            #print(nIndexStart, nIndexEnd, len(sNGSSeq))
            #print(sNGSSeq)
            #sys.exit()

            ### Skip Non-barcodes ###
            try: cPE = dict_sBarcodes[sBarcodeMatch]
            except KeyError:continue
            #########################

            if sBarcodeMatch not in dict_sOutput:
                dict_sOutput[sBarcodeMatch] = []
            dict_sOutput[sBarcodeMatch].append([sReadID, sNGSSeq, nIndexEnd])

            if sBarcodeMatch not in dict_sOutput2:
                dict_sOutput2[sBarcodeMatch] = []
            dict_sOutput2[sBarcodeMatch].append(sSeqData)

        #loop END: i, sReadLine
    #loop END: cPE
    InFile.close()
    print('%s Found= %s' % (sError, len(dict_sOutput)))

    ## Pickle Out ##
    sOutFile = '%s/%s.data' % (sTempOut, sSplitTag)
    OutFile = open(sOutFile, 'wb')
    pickle.dump(dict_sOutput, OutFile)
    OutFile.close()

    sOutFile = '%s/%s.vSeqIO.data' % (sTempOut, sSplitTag)
    OutFile = open(sOutFile, 'wb')
    pickle.dump(dict_sOutput2, OutFile)
    OutFile.close()
#def END: sort_by_barcode


def sort_by_barcode_v2 (list_sParameters):

    print('Sort by Barcode Running - %s' % list_sParameters[1])

    sSplitTag      = list_sParameters[0]
    sInFastq       = list_sParameters[1]
    sTempOut       = list_sParameters[2]
    sBarcodeFile   = list_sParameters[3]
    sRE            = list_sParameters[4]
    sError         = list_sParameters[5]
    sRun           = list_sParameters[6]

    if sRun in ['Offtarget', 'D4D21']:
        nRefBuffer    = 29  #Barcode length to subtract from back of RefSeq  ## For offtarget
        nTargetBuffer = 6   #Barcode length to subtract from front of TarSeq ## For offtarget
    else:
        nRefBuffer     = 30  #Barcode length to subtract from back of RefSeq
        nTargetBuffer  = 4   #Barcode length to subtract from front of TarSeq

    dict_cPE       = load_PE_input_v2(sBarcodeFile)

    dict_sOutput   = {}
    InFile         = open(sInFastq, 'r')
    for i, sReadLine in enumerate(InFile):

        if i % 4 == 0: sReadID = sReadLine.replace('\n', '')
        if i % 4 != 1: continue

        sNGSSeq = sReadLine.replace('\n', '').upper()

        for sReIndex in regex.finditer(sRE, sNGSSeq, overlapped=True):
            nIndexStart   = sReIndex.start()
            nIndexEnd     = sReIndex.end()
            sBarcodeMatch = sNGSSeq[nIndexStart:nIndexEnd]
            sRefSeqCheck  = sNGSSeq[:nIndexStart]

            ### Skip Non-barcodes ###
            try: cPE = dict_cPE[sBarcodeMatch]
            except KeyError: continue
            #########################
            '''
            print('Check', sRefSeqCheck)
            print('NGS  ', cPE.sRefSeq[:-nRefBuffer])
            print()

            print(sNGSSeq)
            print(cPE.sRefSeq)
            print(sBarcodeMatch)
            print('RefCheck-->', sRefSeqCheck)
            print('RefFile -->', cPE.sRefSeq[:-nRefBuffer])
            print()
            '''

            ## Skip error in Refseq ##
            if sError == 'ErrorFree':
                if sRefSeqCheck  != cPE.sRefSeq[:-nRefBuffer]: continue
            ##########################

            if sBarcodeMatch not in dict_sOutput:
                dict_sOutput[sBarcodeMatch] = []
            dict_sOutput[sBarcodeMatch].append([sReadID, sNGSSeq, nIndexEnd-nTargetBuffer])
        #loop END: i, sReadLine
    #loop END: cPE
    InFile.close()

    print('%s Found= %s' % (sError, len(dict_sOutput)))
    ## Pickle Out ##
    sOutFile = '%s/%s.data' % (sTempOut, sSplitTag)
    OutFile = open(sOutFile, 'wb')
    pickle.dump(dict_sOutput, OutFile)
    OutFile.close()
#def END: sort_by_barcode_v2


def sort_by_barcode_v3 (list_sParameters):

    sSplitTag       = list_sParameters[0]
    sInFile         = list_sParameters[1]
    sTempOut        = list_sParameters[2]
    sBarcodeFile    = list_sParameters[3]
    sRE             = list_sParameters[4]
    sError          = list_sParameters[5]
    sRun            = list_sParameters[6]
    nBins           = list_sParameters[7]

    if sRun in ['Offtarget', 'Offtarget2', 'D4D21']:
        nRefBuffer    = 29  #Barcode length to subtract from back of RefSeq  ## For offtarget
        nTargetBuffer = 2   #Barcode length to subtract from front of TarSeq ## For offtarget

    elif sRun in ['Offtarget2-Intended', 'Offtarget2-Intended-Test']:
        nRefBuffer    = 24  #Barcode length to subtract from back of RefSeq  ## For offtarget
        nTargetBuffer = 2   #Barcode length to subtract from front of TarSeq ## For offtarget

    elif sRun in ['Subpool']:
        nRefBuffer     = 0   #Barcode length to subtract from back of RefSeq
        nTargetBuffer  = 0   #Barcode length to subtract from front of TarSeq

    else:
        nRefBuffer     = 30  #Barcode length to subtract from back of RefSeq
        nTargetBuffer  = 0   #Barcode length to subtract from front of TarSeq

    dict_cPE           = load_PE_input_v2(sBarcodeFile)
    list_sBarcodes     = list(dict_cPE.keys())
    nTotalCnt          = len(list_sBarcodes)

    print('Bio.SeqIO Version - Sort by Barcode Running - %s' % (list_sParameters[0]))

    dict_sOutput   = {}
    dict_sOutput2  = {}
    InFile         = open(sInFile, 'r')
    for sSeqData in SeqIO.parse(InFile, 'fastq'):
        sReadID = str(sSeqData.id)
        sNGSSeq = str(sSeqData.seq)

        for sReIndex in regex.finditer(sRE, sNGSSeq, overlapped=True):
            nIndexStart   = sReIndex.start()
            nIndexEnd     = sReIndex.end()
            sBarcodeMatch = sNGSSeq[nIndexStart+nTargetBuffer:nIndexEnd]
            sRefSeqCheck  = sNGSSeq[:nIndexStart]
            sTargetSeq    = sNGSSeq[nIndexEnd:]

            if nRefBuffer == 0: nRefBuffer = (len(sNGSSeq) - nIndexStart - 1)

            ### Skip Non-barcodes ###
            try: cPE = dict_cPE[sBarcodeMatch]
            except KeyError: continue
            #########################
            '''
            if sBarcodeMatch == 'TTTTATACACTCATCACGCGTCTT':
                print(sBarcodeMatch)
                print(nIndexStart, nIndexEnd, len(sNGSSeq), nRefBuffer)
                print(sNGSSeq)
                print(sTargetSeq)
                print(reverse_complement(sTargetSeq))
                sys.exit()
    
                #if cPE.sWTSeq in reverse_complement(sTargetSeq):
                #    print('sWTseq', cPE.sWTSeq)
                #    sys.exit()
                #if cPE.sAltSeq in reverse_complement(sTargetSeq):
                #    print('sAltseq', cPE.sAltSeq)
                #    sys.exit()
            '''
            ## Skip error in Refseq ##
            if sError == 'ErrorFree':
                '''
                print(sNGSSeq)
                print(sRefSeqCheck)
                print(cPE.sRefSeq)
                print(cPE.sRefSeq[:-nRefBuffer])
                print(sBarcodeMatch)
                print()
                if sBarcodeMatch == 'TTTTTCATAGCACGTACATCAGCC': sys.exit()
                '''
                #if sRefSeqCheck  != cPE.sRefSeq[:-nRefBuffer]: continue
                if not cPE.sRefSeq[:-nRefBuffer] in sRefSeqCheck: continue

            ##########################

            if sBarcodeMatch not in dict_sOutput:
                dict_sOutput[sBarcodeMatch] = []
            dict_sOutput[sBarcodeMatch].append([sReadID, sNGSSeq, nIndexEnd-nTargetBuffer])

            if sBarcodeMatch not in dict_sOutput2:
                dict_sOutput2[sBarcodeMatch] = []
            dict_sOutput2[sBarcodeMatch].append(sSeqData)
        #loop END: i, sReadLine
    #loop END: sSeqData
    InFile.close()
    print('%s Found= %s' % (sError, len(dict_sOutput)))

    ## Pickle Out ##
    sOutFile = '%s/%s.data' % (sTempOut, sSplitTag)
    OutFile = open(sOutFile, 'wb')
    pickle.dump(dict_sOutput, OutFile)
    OutFile.close()

    sOutFile = '%s/%s.vSeqIO.data' % (sTempOut, sSplitTag)
    OutFile = open(sOutFile, 'wb')
    pickle.dump(dict_sOutput2, OutFile)
    OutFile.close()

    sTempOut_bybar = '%s/bybarcodes' % sTempOut
    os.makedirs(sTempOut_bybar, exist_ok=True)

    ## Output By Barcode ##
    for sBarcode in dict_sOutput2:
        sOutFile = '%s/%s.fastq' % (sTempOut_bybar, sBarcode)
        OutFile  = open(sOutFile, 'w')

        #sOut     = '%s\t%s\n' % (sBarcode, ','.join(dict_sOutput2[sBarcode]))
        #OutFile.write(sOut)

        for sSeqData in dict_sOutput2[sBarcode]:
            try: SeqIO.write(sSeqData, OutFile, 'fastq')
            except TypeError: continue
        #loop END: sReadID, sNGSRead

        OutFile.close()
    #loop END: sBarcode
    print('Bio.SeqIO Version - Sort by Barcode DONE - %s' % (list_sParameters[0]))
#def END: sort_by_barcode_v3


def sort_by_barcode_vOfftarget3 (list_sParameters):

    print('Processing Offtarget3 %s' % list_sParameters[1])

    sSplitTag      = list_sParameters[0]
    sInFastq       = list_sParameters[1]
    sTempOut       = list_sParameters[2]
    sBarcodeFile   = list_sParameters[3]
    sRE            = list_sParameters[4]
    sError         = list_sParameters[5]
    sOutDir        = list_sParameters[-1]

    nBarcode3Cut    = 3  #
    nRefBuffer      = 21  # Barcode length to subtract from back of RefSeq
    nTargetBuffer   = 4  # Barcode length to subtract from front of TarSeq

    dict_sBarcodes = load_PE_input_v3(sBarcodeFile)

    print('Bio.SeqIO Version - Sort by Barcode Running - %s' % (list_sParameters[0]))

    dict_sOutput   = {}
    dict_sOutput2  = {}
    InFile         = open(sInFastq, 'r')

    ## Check Loss ##
    dict_sCheckLoss = {}
    ################

    nCnt1 = 0
    nCnt2 = 0
    for sSeqData in SeqIO.parse(InFile, 'fastq'):

        sReadID = str(sSeqData.id)
        sNGSSeq = str(sSeqData.seq)

        for sReIndex in regex.finditer(sRE, sNGSSeq, overlapped=True):
            nIndexStart   = sReIndex.start()
            nIndexEnd     = sReIndex.end()
            sBarcodeMatch = sNGSSeq[nIndexStart+nBarcode3Cut:nIndexEnd] #if RE = [T]{4}
            #sBarcode     = sNGSSeq[nIndexStart:nIndexEnd]
            sRefSeqCheck  = sNGSSeq[:nIndexStart]

            ### Skip Non-barcodes ###
            try: cPE = dict_sBarcodes[sBarcodeMatch]
            except KeyError:continue

            #########################
            ### Skip error in Refseq ###
            if sError == 'ErrorFree':
                #if sRefSeqCheck  != cPE.sRefSeq_conv[:-nRefBuffer] and sRefSeqCheck  != cPE.sRefSeq_opti[:-nRefBuffer]:
                if cPE.sRefSeq_conv[:-nRefBuffer] not in sRefSeqCheck and  cPE.sRefSeq_opti[:-nRefBuffer] not in sRefSeqCheck:
                    if sBarcodeMatch not in dict_sCheckLoss:
                        dict_sCheckLoss[sBarcodeMatch] = []
                    dict_sCheckLoss[sBarcodeMatch].append(sSeqData)

                    continue
            ##########################
            '''
            print('sBarcodeMatch', sBarcodeMatch)
            print(sNGSSeq)
            print(cPE.sRefSeq_conv)
            print(cPE.sRefSeq_opti)

            print('RefCheck-->', sRefSeqCheck)
            print('RefFile Conv -->', cPE.sRefSeq_conv[:-nRefBuffer], sRefSeqCheck == cPE.sRefSeq_conv[:-nRefBuffer])
            print('RefFile Opti-->', cPE.sRefSeq_opti[:-nRefBuffer], sRefSeqCheck == cPE.sRefSeq_opti[:-nRefBuffer])
            print()
            '''

            if sBarcodeMatch not in dict_sOutput:
                dict_sOutput[sBarcodeMatch] = []
            dict_sOutput[sBarcodeMatch].append([sReadID, sNGSSeq, nIndexEnd])

            if sBarcodeMatch not in dict_sOutput2:
                dict_sOutput2[sBarcodeMatch] = []
            dict_sOutput2[sBarcodeMatch].append(sSeqData)

        #loop END: i, sReadLine
    #loop END: cPE
    InFile.close()
    print('%s Found= %s' % (sError, len(dict_sOutput)))
    #print(nCnt1, nCnt2)

    ## Pickle Out ##
    sOutFile = '%s/%s.data' % (sTempOut, sSplitTag)
    OutFile = open(sOutFile, 'wb')
    pickle.dump(dict_sOutput, OutFile)
    OutFile.close()

    sOutFile = '%s/%s.vSeqIO.data' % (sTempOut, sSplitTag)
    OutFile = open(sOutFile, 'wb')
    pickle.dump(dict_sOutput2, OutFile)
    OutFile.close()

    ## Output By Barcode for Check Loss ##
    sCheckLossDir   = '%s/checkloss/%s' % (sOutDir, sSplitTag)
    os.makedirs(sCheckLossDir, exist_ok=True)

    for sBarcode in dict_sCheckLoss:
        sOutFile = '%s/%s.fastq' % (sCheckLossDir, sBarcode)
        OutFile = open(sOutFile, 'w')

        # sOut     = '%s\t%s\n' % (sBarcode, ','.join(dict_sOutput2[sBarcode]))
        # OutFile.write(sOut)

        for sSeqData in dict_sCheckLoss[sBarcode]:
            try:
                SeqIO.write(sSeqData, OutFile, 'fastq')
            except TypeError:
                continue
        # loop END: sReadID, sNGSRead

        OutFile.close()
    # loop END: sBarcode
#def END: sort_by_barcode


def determine_output (list_sParameters):

    print('Processing %s' % list_sParameters[2])
    sSplitTag      = list_sParameters[0]
    sInFastq       = list_sParameters[1]
    sTempOut       = list_sParameters[2]
    sBarcodeFile   = list_sParameters[3]
    dict_cPE       = load_PE_input(sBarcodeFile)

    ## Pickle Load ##
    sInFile        = '%s/%s.data' % (sTempOut, sSplitTag)
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

        for sReadID, sNGSSeq in dict_sBarcodes[sBarcode]:

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
            #if END:
        #loop END: sReadID, sNGSSeq
    #loop END: sBarcode
    list_sKeys = ['WT', 'Alt', 'Other']

    sOutFile   = '%s/%s.reads.txt' % (sTempOut, sSplitTag)
    OutFile    = open(sOutFile, 'w')

    for sBarcode in dict_sOutput:
        sOut = '%s\t%s\n' % (sBarcode, '\t'.join([','.join(dict_sOutput[sBarcode][sType]) for sType in list_sKeys]))
        OutFile.write(sOut)
    # loop END: sBarcode
    OutFile.close()
#def END: determine_output


def determine_output_v2 (list_sParameters):

    print('Determine Output - %s' % list_sParameters[2])

    sSplitTag      = list_sParameters[0]
    sInFastq       = list_sParameters[1]
    sTempOut       = list_sParameters[2]
    sBarcodeFile   = list_sParameters[3]
    dict_cPE       = load_PE_input_v2(sBarcodeFile)

    ## Pickle Load ##
    sInFile        = '%s/%s.data' % (sTempOut, sSplitTag)
    InFile         = open(sInFile, 'rb')
    dict_sBarcodes = pickle.load(InFile)
    InFile.close()
    print('%s dict_sBarcodes= %s' % (sSplitTag, len(dict_sBarcodes)))

    dict_sOutput   = {}
    for sBarcode in dict_sBarcodes:

        cPE      = dict_cPE[sBarcode]

        if sBarcode not in dict_sOutput:
            dict_sOutput[sBarcode] = {'WT': [], 'Alt': [], 'Other': []}

        nWTSize  = len(cPE.sWTSeq)
        nAltSize = len(cPE.sAltSeq)

        for sReadID, sNGSSeq, nIndexS in dict_sBarcodes[sBarcode]:

            nAdjusted      = nIndexS - 4

            sWTSeqCheck    = sNGSSeq[nAdjusted:nAdjusted+nWTSize]
            sAltSeqCheck   = sNGSSeq[nAdjusted:nAdjusted+nAltSize]

            nDiff = abs(len(sWTSeqCheck) - len(sAltSeqCheck))

            if sWTSeqCheck == cPE.sWTSeq:
                dict_sOutput[cPE.sBarcode]['WT'].append(sWTSeqCheck)

            elif sAltSeqCheck == cPE.sAltSeq:
                dict_sOutput[cPE.sBarcode]['Alt'].append(sAltSeqCheck)

            elif sWTSeqCheck != cPE.sWTSeq and sAltSeqCheck != cPE.sAltSeq:
                dict_sOutput[cPE.sBarcode]['Other'].append(sAltSeqCheck)
            #if END:
        #loop END: sReadID, sNGSSeq

    #loop END: sBarcode
    list_sKeys = ['WT', 'Alt', 'Other']

    sOutFile   = '%s/%s.reads.txt' % (sTempOut, sSplitTag)
    OutFile    = open(sOutFile, 'w')

    for sBarcode in dict_sOutput:
        sOut = '%s\t%s\n' % (sBarcode, '\t'.join([','.join(dict_sOutput[sBarcode][sType]) for sType in list_sKeys]))
        OutFile.write(sOut)
    # loop END: sBarcode
    OutFile.close()
#def END: determine_output_v2


def determine_output_vOfftarget2 (list_sParameters): #For offtarget2

    print('Determine Output - %s' % list_sParameters[2])

    sSplitTag      = list_sParameters[0]
    sInFastq       = list_sParameters[1]
    sTempOut       = list_sParameters[2]
    sBarcodeFile   = list_sParameters[3]
    dict_cPE       = load_PE_input_v2(sBarcodeFile)

    ## Pickle Load ##
    sInFile        = '%s/%s.data' % (sTempOut, sSplitTag)
    InFile         = open(sInFile, 'rb')
    dict_sBarcodes = pickle.load(InFile)
    InFile.close()
    print('%s dict_sBarcodes= %s' % (sSplitTag, len(dict_sBarcodes)))

    dict_sOutput   = {}
    for sBarcode in dict_sBarcodes:

        #if sBarcode != 'TTTTCACACTCATCAGCTCACGCC': continue

        cPE     = dict_cPE[sBarcode]

        if sBarcode not in dict_sOutput:
            dict_sOutput[sBarcode] = {'WT': [], 'Alt': [], 'Other': []}

        nWTSize  = len(cPE.sWTSeq)
        nAltSize = len(cPE.sAltSeq)

        #print(cPE.sWTSeq)
        #print(cPE.sAltSeq)

        for sReadID, sNGSSeq, sIndexS in dict_sBarcodes[sBarcode]:

            sTargetSeq = reverse_complement(sNGSSeq[sIndexS-1:])
            #print(sNGSSeq)
            #print(sTargetSeq)

            if cPE.sWTSeq in sTargetSeq:
                dict_sOutput[cPE.sBarcode]['WT'].append(cPE.sWTSeq)

            elif cPE.sAltSeq in sTargetSeq:
                dict_sOutput[cPE.sBarcode]['Alt'].append(cPE.sAltSeq)

            elif cPE.sWTSeq not in sTargetSeq and cPE.sAltSeq not in sTargetSeq:
                dict_sOutput[cPE.sBarcode]['Other'].append(cPE.sAltSeq)
            #if END:
        #loop END: sReadID, sNGSSeq

    #loop END: sBarcode
    list_sKeys = ['WT', 'Alt', 'Other']

    sOutFile   = '%s/%s.reads.txt' % (sTempOut, sSplitTag)
    OutFile    = open(sOutFile, 'w')

    for sBarcode in dict_sOutput:
        #sOut = '%s\t%s\n' % (sBarcode, '\t'.join([str(len(dict_sOutput[sBarcode][sType])) for sType in list_sKeys]))
        sOut = '%s\t%s\n' % (sBarcode, '\t'.join([','.join(dict_sOutput[sBarcode][sType]) for sType in list_sKeys]))
        #print(sOut[:-1])
        OutFile.write(sOut)
    # loop END: sBarcode
    OutFile.close()
#def END: determine_output_vOfftarget2


def determine_output_vOfftarget2_Intended (list_sParameters): #For offtarget2, intended and mismatch

    print('Determine Output - %s' % list_sParameters[2])

    sSplitTag      = list_sParameters[0]
    sInFastq       = list_sParameters[1]
    sTempOut       = list_sParameters[2]
    sBarcodeFile   = list_sParameters[3]
    dict_cPE       = load_PE_input_v2(sBarcodeFile)

    ## Pickle Load ##
    sInFile        = '%s/%s.data' % (sTempOut, sSplitTag)
    InFile         = open(sInFile, 'rb')
    dict_sBarcodes = pickle.load(InFile)
    InFile.close()
    print('%s dict_sBarcodes= %s' % (sSplitTag, len(dict_sBarcodes)))

    dict_sOutput   = {}
    for sBarcode in dict_sBarcodes:

        cPE     = dict_cPE[sBarcode]

        if sBarcode not in dict_sOutput:
            dict_sOutput[sBarcode] = {'WT': [], 'Alt': [], 'Intended':[], 'Mismatch':[], 'Other': []}

        nWTSize  = len(cPE.sWTSeq)
        nAltSize = len(cPE.sAltSeq)

        for sReadID, sNGSSeq, sIndexS in dict_sBarcodes[sBarcode]:

            sTargetSeq = reverse_complement(sNGSSeq[sIndexS:])
            '''
            if sBarcode == 'TTTTCGTCACACTATCACATACCC':
                print(sBarcode)
                print(sNGSSeq)
                print(sTargetSeq)
                print(reverse_complement(sTargetSeq))
                print(cPE.sWTSeq)
                print(cPE.sAltSeq)
                print(cPE.sIntendedOnly)
                print(cPE.sMisMatchOnly)
                sys.exit()
                #if cPE.sWTSeq in reverse_complement(sTargetSeq):
                #    print('sWTseq', cPE.sWTSeq)
                #    sys.exit()
                #if cPE.sAltSeq in reverse_complement(sTargetSeq):
                #    print('sAltseq', cPE.sAltSeq)
                #    sys.exit()

                if cPE.sWTSeq in sTargetSeq:
                    print(sNGSSeq)
                    print(sTargetSeq)
                    print('sWTSeq', cPE.sWTSeq)
                    sys.exit()

                if cPE.sAltSeq in sTargetSeq:
                    print('sAltSeq', cPE.sAltSeq)
                    print(sNGSSeq)
                    print(sTargetSeq)
                    sys.exit()
            '''

            #dict_sCheckWT = get_key_window (sTargetSeq, len(cPE.sWTSeq))


            if cPE.sWTSeq in sTargetSeq:
                dict_sOutput[cPE.sBarcode]['WT'].append(cPE.sWTSeq)

            elif cPE.sAltSeq in sTargetSeq:
                dict_sOutput[cPE.sBarcode]['Alt'].append(cPE.sAltSeq)

            elif cPE.sIntendedOnly in sTargetSeq:
                dict_sOutput[cPE.sBarcode]['Intended'].append(cPE.sAltSeq)

            elif cPE.sMisMatchOnly in sTargetSeq:
                dict_sOutput[cPE.sBarcode]['Mismatch'].append(cPE.sAltSeq)

            elif cPE.sWTSeq not in sTargetSeq and cPE.sAltSeq not in sTargetSeq\
                    and cPE.sIntendedOnly not in sTargetSeq and cPE.sMisMatchOnly not in sTargetSeq:
                dict_sOutput[cPE.sBarcode]['Other'].append(cPE.sAltSeq)
            #if END:
        #loop END: sReadID, sNGSSeq
    #loop END: sBarcode
    list_sKeys = ['WT', 'Alt', 'Intended', 'Mismatch', 'Other']

    sOutFile   = '%s/%s.reads.txt' % (sTempOut, sSplitTag)
    OutFile    = open(sOutFile, 'w')

    for sBarcode in dict_sOutput:
        sOut1 = '%s\t%s\n' % (sBarcode, '\t'.join([str(len(dict_sOutput[sBarcode][sType])) for sType in list_sKeys]))
        sOut2 = '%s\t%s\n' % (sBarcode, '\t'.join([','.join(dict_sOutput[sBarcode][sType]) for sType in list_sKeys]))
        #print(sOut1[:-1])
        OutFile.write(sOut2)
    # loop END: sBarcode
    OutFile.close()
#def END: determine_output_vOfftarget2_Intended


def get_key_window (sTargetSeq, nWindow):

    dict_sOutput = {}
    nTotal       = len(sTargetSeq)
    print(nTotal - nWindow)
    for i in range(0, (nTotal-nWindow)+1):

        sSeq = sTargetSeq[i:i+nWindow]
        print(sSeq, i)
        if sSeq not in dict_sOutput:
            dict_sOutput[sSeq] = 0
        dict_sOutput[sSeq] += 1
        #if i == (nTotal-nWindow): break

    sys.exit()


#def END: get_key_window




def determine_output_vOfftarget3 (list_sParameters): #For offtarget2

    print('Determine Output - %s' % list_sParameters[2])

    sSplitTag      = list_sParameters[0]
    sInFastq       = list_sParameters[1]
    sTempOut       = list_sParameters[2]
    sBarcodeFile   = list_sParameters[3]
    dict_cPE       = load_PE_input_v3(sBarcodeFile)

    #from Offtarget3
    list_sScaffWindow = [35, 65]
    sScaffSeq_conv    = 'TAGAGCTAGAAATAG'
    sScaffSeq_opti    = 'CAGAGCTATGCTGGA'

    ## Pickle Load ##
    sInFile        = '%s/%s.data' % (sTempOut, sSplitTag)
    InFile         = open(sInFile, 'rb')
    dict_sBarcodes = pickle.load(InFile)
    InFile.close()
    print('%s dict_sBarcodes= %s' % (sSplitTag, len(dict_sBarcodes)))

    dict_sOutput   = {}
    for sBarcode in dict_sBarcodes:

        #if sBarcode != 'TTTTTGTGTACATCTGTGCC': continue

        cPE     = dict_cPE[sBarcode]

        if sBarcode not in dict_sOutput:
            dict_sOutput[sBarcode] = {'conv':{'WT': [], 'Alt': [], 'Other': []},
                                      'opti': {'WT': [], 'Alt': [], 'Other': []},
                                      'noscaff': {'WT': [], 'Alt': [], 'Other': []},
                                      }

        nWTSize  = len(cPE.sWTSeq)
        nAltSize = len(cPE.sAltSeq)
        #print('WT', nWTSize, cPE.sWTSeq)
        #print('Alt',nAltSize,  cPE.sAltSeq)

        for sReadID, sNGSSeq, sIndexS in dict_sBarcodes[sBarcode]:

            sScaffCheck   = sNGSSeq[list_sScaffWindow[0]:list_sScaffWindow[1]]
            sTargetCheck  = reverse_complement(sNGSSeq[sIndexS-1:])

            print('NGS', sNGSSeq)
            print(sScaffCheck)
            print(sTargetCheck)
            sys.exit()

            if sScaffSeq_conv in sScaffCheck:

                if cPE.sWTSeq in sTargetCheck:
                    dict_sOutput[cPE.sBarcode]['conv']['WT'].append(cPE.sWTSeq)

                elif cPE.sAltSeq in sTargetCheck:
                    dict_sOutput[cPE.sBarcode]['conv']['Alt'].append(cPE.sAltSeq)

                elif cPE.sWTSeq not in sTargetCheck and cPE.sAltSeq not in sTargetCheck:
                    dict_sOutput[cPE.sBarcode]['conv']['Other'].append(cPE.sAltSeq)

            elif sScaffSeq_opti in sScaffCheck:
                if cPE.sWTSeq in sTargetCheck:
                    dict_sOutput[cPE.sBarcode]['opti']['WT'].append(cPE.sWTSeq)

                elif cPE.sAltSeq in sTargetCheck:
                    dict_sOutput[cPE.sBarcode]['opti']['Alt'].append(cPE.sAltSeq)

                elif cPE.sWTSeq not in sTargetCheck and cPE.sAltSeq not in sTargetCheck:
                    dict_sOutput[cPE.sBarcode]['opti']['Other'].append(cPE.sAltSeq)
            else:
                if cPE.sWTSeq in sTargetCheck:
                    dict_sOutput[cPE.sBarcode]['noscaff']['WT'].append(cPE.sWTSeq)

                elif cPE.sAltSeq in sTargetCheck:
                    dict_sOutput[cPE.sBarcode]['noscaff']['Alt'].append(cPE.sAltSeq)

                elif cPE.sWTSeq not in sTargetCheck and cPE.sAltSeq not in sTargetCheck:
                    dict_sOutput[cPE.sBarcode]['noscaff']['Other'].append(cPE.sAltSeq)
            #if END:
        #loop END: sReadID, sNGSSeq
    #loop END: sBarcode

    list_sScaffKeys = ['conv', 'opti', 'noscaff']
    list_sTypeKeys  = ['WT', 'Alt', 'Other']


    sOutFile   = '%s/%s.reads.txt' % (sTempOut, sSplitTag)
    OutFile    = open(sOutFile, 'w')

    dict_sOutput2 = {}
    dict_sOutput2['test'] = {'conv': {'WT': ['conv-wt'], 'Alt': ['conv-alt'], 'Other': ['conv-other']},
                              'opti': {'WT': ['opti-wt'], 'Alt': ['opti-alt'], 'Other': ['opti-other']},
                              'noscaff': {'WT': ['noscaff-wt'], 'Alt': ['noscaff-alt'], 'Other': ['noscaff-other']},
                              }

    for sBarcode in dict_sOutput:
        #sOut = '%s\t%s\n' % (sBarcode, '\t'.join([str(len(dict_sOutput[sBarcode][sScaffType][sAltType])) for sScaffType in list_sScaffKeys for sAltType in list_sTypeKeys]))
        sOut = '%s\t%s\n' % (sBarcode, '\t'.join([','.join(dict_sOutput[sBarcode][sScaffType][sAltType]) for sScaffType in list_sScaffKeys for sAltType in list_sTypeKeys]))
        #print(sOut[:-1])
        OutFile.write(sOut)
    # loop END: sBarcode
    OutFile.close()
#def END: determine_output_vOfftarget3


def determine_output_vJustNGS(list_sParameters):

    sSplitTag         = list_sParameters[0]
    sTempDir          = list_sParameters[1]
    sNGSOut           = list_sParameters[2]
    sBarcodeFile      = list_sParameters[3]
    sCntTag           = list_sParameters[4]

    dict_cPE          = load_PE_input_v2(sBarcodeFile)
    list_sTarBars     = list(dict_cPE.keys())

    ## Pickle Load ##
    sInFile           = '%s/%s.vSeqIO.data' % (sTempDir, sSplitTag)
    InFile            = open(sInFile, 'rb')
    dict_sBarcodes    = pickle.load(InFile)
    InFile.close()
    print('%s %s %s dict_sBarcodes= %s' % (sCntTag, sSplitTag, sNGSOut, len(dict_sBarcodes)))

    list_sBarcodes    = [sBarcode for sBarcode in dict_sBarcodes if sBarcode in set(list_sTarBars)]

    for sBarcode in list_sBarcodes:
        sOutFile = '%s/%s.justNGS.txt' % (sNGSOut, sBarcode)
        OutFile  = open(sOutFile, 'w')

        for sSeqData in dict_sBarcodes[sBarcode]:

            try: SeqIO.write(sSeqData, OutFile, 'fastq')
            except TypeError: continue
        #loop END: sReadID, sNGSRead
        OutFile.close()
    #loop END: sBarcode
# def END: determine_output_vJustNGS


def combine_output_pickle (sSample, sOutDir, sRun, sError, sBarcodeFile, list_sSplitFiles, nSplitNo):

    dict_cPE      = load_PE_input_v2(sBarcodeFile)
    sTempDir      = '%s/%s/temp'     % (sOutDir, sError)
    '''
    if sRun in ['Offtarget3']:

        list_sKeys = ['WT-conv', 'Alt-conv', 'Other-conv','WT-opti',
                      'Alt-opti', 'Other-opti', 'WT-none', 'Alt-none', 'Other-none',]

    elif sRun in ['Offtarget2-Intended']:

        list_sKeys = ['WT', 'Alt', 'Intended', 'Mismatch', 'Other']

    else:
        list_sKeys = ['WT', 'Alt', 'Other']
    '''

    if sRun in ['Offtarget3']:
        temp_output_in_parts_vOfftarget3 (sTempDir, dict_cPE, list_sSplitFiles, sSample, nSplitNo)

    elif sRun in ['Offtarget2-Intended', 'Offtarget2-Intended-Test']:
        temp_output_in_parts_vOfftarget2_Intended (sTempDir, dict_cPE, list_sSplitFiles, sSample, nSplitNo)

    else:
        temp_output_in_parts (sTempDir, dict_cPE, list_sSplitFiles, sSample, nSplitNo)

#def END: combine_output


def combine_output_freq (sSample, sOutputDir, sRun, sError, sBarcodeFile, list_sSplitFiles, nSplitNo):

    print('combine_output_freq - %s' % sSample)
    sOutDir       = '%s/%s' % (sOutputDir, sError)
    os.makedirs(sOutDir, exist_ok=True)

    dict_cPE      = load_PE_input_v2(sBarcodeFile)
    sCombineOut   = '%s/combinedFreq' % sOutDir
    sOthersOut    = '%s/others'       % sOutDir
    sTempDir      = '%s/temp'         % sOutDir

    os.makedirs(sCombineOut, exist_ok=True)
    os.makedirs(sOthersOut, exist_ok=True)

    if sRun in ['Offtarget3']:

        list_sKeys = ['WT-conv', 'Alt-conv', 'Other-conv','WT-opti',
                      'Alt-opti', 'Other-opti', 'WT-none', 'Alt-none', 'Other-none',]

    elif sRun in ['Offtarget2-Intended', 'Offtarget2-Intended-Test']:
        list_sKeys = ['WT', 'Alt', 'Intended', 'Mismatch', 'Other']

    else:
        list_sKeys = ['WT', 'Alt', 'Other']
    #if END:

    dict_sOutFreq  = {sBarcode: [0 for i in list_sKeys] for sBarcode in dict_cPE}
    nFileCnt       = len(list_sSplitFiles)
    list_nBins     = [[int(nFileCnt * (i + 0) / nSplitNo), int(nFileCnt * (i + 1) / nSplitNo)] for i in range(nSplitNo)]

    for nStart, nEnd in list_nBins:

        sInFile  = '%s/%s.TempFreq%s-%s.data'    % (sTempDir, sSample, nStart, nEnd)
        InFile   = open(sInFile, 'rb')
        dict_sOutFreq_Temp = pickle.load(InFile)
        InFile.close()
        for sBarcode in dict_cPE:

            if sRun in ['Offtarget3']:
                dict_sOutFreq[sBarcode][0] += dict_sOutFreq_Temp[sBarcode][0]
                dict_sOutFreq[sBarcode][1] += dict_sOutFreq_Temp[sBarcode][1]
                dict_sOutFreq[sBarcode][2] += dict_sOutFreq_Temp[sBarcode][2]
                dict_sOutFreq[sBarcode][3] += dict_sOutFreq_Temp[sBarcode][3]
                dict_sOutFreq[sBarcode][4] += dict_sOutFreq_Temp[sBarcode][4]
                dict_sOutFreq[sBarcode][5] += dict_sOutFreq_Temp[sBarcode][5]
                dict_sOutFreq[sBarcode][6] += dict_sOutFreq_Temp[sBarcode][6]
                dict_sOutFreq[sBarcode][7] += dict_sOutFreq_Temp[sBarcode][7]
                dict_sOutFreq[sBarcode][8] += dict_sOutFreq_Temp[sBarcode][8]

            elif sRun in ['Offtarget2-Intended', 'Offtarget2-Intended-Test']:
                dict_sOutFreq[sBarcode][0] += dict_sOutFreq_Temp[sBarcode][0]
                dict_sOutFreq[sBarcode][1] += dict_sOutFreq_Temp[sBarcode][1]
                dict_sOutFreq[sBarcode][2] += dict_sOutFreq_Temp[sBarcode][2]
                dict_sOutFreq[sBarcode][3] += dict_sOutFreq_Temp[sBarcode][3]
                dict_sOutFreq[sBarcode][4] += dict_sOutFreq_Temp[sBarcode][4]

            else:
                dict_sOutFreq[sBarcode][0] += dict_sOutFreq_Temp[sBarcode][0]
                dict_sOutFreq[sBarcode][1] += dict_sOutFreq_Temp[sBarcode][1]
                dict_sOutFreq[sBarcode][2] += dict_sOutFreq_Temp[sBarcode][2]

        #loop END: sBarcode
    #loop END: nStart, nEnd

    sHeader   = '%s\t%s\t%s\n' % ('Barcode', '\t'.join(list_sKeys), 'Total')
    sOutFile  = '%s/%s.combinedFreq.txt'    % (sCombineOut, sSample)
    OutFile   = open(sOutFile, 'w')
    OutFile.write(sHeader)
    for sBarcode in dict_cPE:

        list_sOut = [str(sOutput) for sOutput in dict_sOutFreq[sBarcode]]
        nTotal  = sum(dict_sOutFreq[sBarcode])
        sOut    = '%s\t%s\t%s\n' % (sBarcode, '\t'.join(list_sOut), nTotal)
        OutFile.write(sOut)
    #loop END: sBarcode
    OutFile.close()

#def END: combine_output


def combine_output_reads (sSample, sOutputDir, sError, sBarcodeFile, list_sSplitFiles, nSplitNo):

    print('combine_output_reads - %s' % sSample)

    sOutDir       = '%s/%s' % (sOutputDir, sError)
    os.makedirs(sOutDir, exist_ok=True)

    dict_cPE      = load_PE_input_v2(sBarcodeFile)
    sCombineOut   = '%s/combinedReads' % sOutDir
    sOthersOut    = '%s/others'        % sOutDir
    sTempDir      = '%s/temp'          % sOutDir

    os.makedirs(sCombineOut, exist_ok=True)
    os.makedirs(sOthersOut, exist_ok=True)

    dict_sAltSeqs  = {sBarcode: [] for sBarcode in dict_cPE}
    dict_sOthers   = {sBarcode: [] for sBarcode in dict_cPE}
    nFileCnt       = len(list_sSplitFiles)
    list_nBins     = [[int(nFileCnt * (i + 0) / nSplitNo), int(nFileCnt * (i + 1) / nSplitNo)] for i in range(nSplitNo)]

    for nStart, nEnd in list_nBins:

        sInFile  = '%s/%s.TempReads%s-%s.data'    % (sTempDir, sSample, nStart, nEnd)
        InFile   = open(sInFile, 'rb')
        dict_sOutReads_Temp = pickle.load(InFile)
        InFile.close()
        for sBarcode in dict_cPE:
            dict_sAltSeqs[sBarcode] += dict_sOutReads_Temp[sBarcode][1]
            dict_sOthers[sBarcode]  += dict_sOutReads_Temp[sBarcode][2]
        #loop END: sBarcode
    #loop END: nStart, nEnd

    sOutDir    = '%s/%s' % (sOthersOut, sSample)
    os.makedirs(sOutDir, exist_ok=True)

    ## Alt Out ##
    sOutFile  = '%s/%s.AltReads.txt' % (sCombineOut, sSample)
    OutFile   = open(sOutFile, 'w')

    ## Others Out ##
    sOutFile2  = '%s/%s.OtherReads.txt' % (sCombineOut, sSample)
    OutFile2   = open(sOutFile2, 'w')

    for sBarcode in dict_cPE:

        dict_sAltOut        = groupby_element (dict_sAltSeqs[sBarcode])
        dict_sOtherOut      = groupby_element (dict_sOthers[sBarcode])

        list_sAltReads      = ['%s:%s' % (sBarcode, dict_sAltOut[sBarcode]) for sBarcode in dict_sAltOut]
        list_sOtherReads    = ['%s:%s' % (sBarcode, dict_sOtherOut[sBarcode]) for sBarcode in dict_sOtherOut]

        sOut                = '%s\t%s\n' % (sBarcode, ','.join(list_sAltReads))
        sOut2               = '%s\t%s\n' % (sBarcode, ','.join(list_sOtherReads))
        OutFile.write(sOut)
        OutFile2.write(sOut2)
    #loop END: sBarcode
    OutFile.close()
    OutFile2.close()
#def END: combine_output


def temp_output_in_parts (sTempDir, dict_cPE, list_sSplitFiles, sSample, nNoSplits):

    nFileCnt   = len(list_sSplitFiles)
    list_nBins = [[int(nFileCnt * (i + 0) / nNoSplits), int(nFileCnt * (i + 1) / nNoSplits)] for i in range(nNoSplits)]

    for nStart, nEnd in list_nBins:
        list_sSubSplit   = list_sSplitFiles[nStart:nEnd]
        dict_sOutFreq_H  = {sBarcode: [0, 0, 0]    for sBarcode in dict_cPE}
        dict_sOutRead_H  = {sBarcode: [[], [], []] for sBarcode in dict_cPE}

        for i, sSplitFile in enumerate(list_sSubSplit):
            print('Output Temp Pickle %s/%s -- %s' % ((i+nStart), nEnd, sSplitFile))
            sSplitTag  = '_'.join(sSplitFile.split('.')[1].split('_')[-2:])
            sTempFile  = '%s/%s/%s'         % (sTempDir, sSample, sSplitTag)
            sInFile    = '%s/%s.reads.txt'  % (sTempFile, sSplitTag)
            print(sInFile)

            assert os.path.isfile(sInFile)
            load_read_data (dict_sOutFreq_H, dict_sOutRead_H, sInFile)
        #loop END: sSplitFile

        sOutFile  = '%s/%s.TempFreq%s-%s.data'    % (sTempDir, sSample, nStart, nEnd)
        OutFile  = open(sOutFile, 'wb')
        pickle.dump(dict_sOutFreq_H, OutFile)
        OutFile.close()

        sOutFile  = '%s/%s.TempReads%s-%s.data'    % (sTempDir, sSample, nStart, nEnd)
        OutFile  = open(sOutFile, 'wb')
        pickle.dump(dict_sOutRead_H, OutFile)
        OutFile.close()
    #loop END: nStart, nEnd
#def END: temp_output_in_parts


def load_read_data (dict_sOutFreq, dict_sOutRead, sInFile):

    InFile       = open(sInFile, 'r')
    for sReadLine in InFile:

        list_sColumn      = sReadLine.strip('\n').split('\t')
        sBarcode          = list_sColumn[0]
        list_WT_reads     = [sReadID for sReadID in list_sColumn[1].split(',') if sReadID]
        list_edited_reads = [sReadID for sReadID in list_sColumn[2].split(',') if sReadID]
        list_other_reads  = [sReadID for sReadID in list_sColumn[3].split(',') if sReadID]

        dict_sOutFreq[sBarcode][0] += len(list_WT_reads)
        dict_sOutFreq[sBarcode][1] += len(list_edited_reads)
        dict_sOutFreq[sBarcode][2] += len(list_other_reads)

        dict_sOutRead[sBarcode][0] += list_WT_reads
        dict_sOutRead[sBarcode][1] += list_edited_reads
        dict_sOutRead[sBarcode][2] += list_other_reads
    #loop END: sReadLine
    InFile.close()
#def END: load_freq_data


def temp_output_in_parts_vOfftarget3 (sTempDir, dict_cPE, list_sSplitFiles, sSample, nNoSplits):

    nFileCnt   = len(list_sSplitFiles)
    list_nBins = [[int(nFileCnt * (i + 0) / nNoSplits), int(nFileCnt * (i + 1) / nNoSplits)] for i in range(nNoSplits)]

    for nStart, nEnd in list_nBins:
        list_sSubSplit   = list_sSplitFiles[nStart:nEnd]
                            #conv, opti, and no scafford (WT, ALT, Others for each)
        dict_sOutFreq_H  = {sBarcode: [0, 0, 0, 0, 0, 0, 0, 0, 0] for sBarcode in dict_cPE}
        dict_sOutRead_H  = {sBarcode: [[], [], [], [], [], [], [], [], []] for sBarcode in dict_cPE}

        for i, sSplitFile in enumerate(list_sSubSplit):
            print('Output Temp Pickle %s/%s -- %s' % ((i+nStart), nEnd, sSplitFile))
            sSplitTag  = '_'.join(sSplitFile.split('.')[1].split('_')[-2:])
            sTempFile  = '%s/%s/%s'         % (sTempDir, sSample, sSplitTag)
            sInFile    = '%s/%s.reads.txt'  % (sTempFile, sSplitTag)
            print(sInFile)

            assert os.path.isfile(sInFile)
            load_read_data_vOfftarget3 (dict_sOutFreq_H, dict_sOutRead_H, sInFile)
        #loop END: sSplitFile

        sOutFile  = '%s/%s.TempFreq%s-%s.data'    % (sTempDir, sSample, nStart, nEnd)
        OutFile  = open(sOutFile, 'wb')
        pickle.dump(dict_sOutFreq_H, OutFile)
        OutFile.close()

        sOutFile  = '%s/%s.TempReads%s-%s.data'    % (sTempDir, sSample, nStart, nEnd)
        OutFile  = open(sOutFile, 'wb')
        pickle.dump(dict_sOutRead_H, OutFile)
        OutFile.close()
    #loop END: nStart, nEnd
#def END: temp_output_in_parts


def load_read_data_vOfftarget3 (dict_sOutFreq, dict_sOutRead, sInFile):

    InFile       = open(sInFile, 'r')
    for sReadLine in InFile:

        list_sColumn      = sReadLine.strip('\n').split('\t')
        sBarcode          = list_sColumn[0]

        list_WT_conv      = [sReadID for sReadID in list_sColumn[1].split(',') if sReadID]
        list_edited_conv  = [sReadID for sReadID in list_sColumn[2].split(',') if sReadID]
        list_other_conv   = [sReadID for sReadID in list_sColumn[3].split(',') if sReadID]

        dict_sOutFreq[sBarcode][0] += len(list_WT_conv)
        dict_sOutFreq[sBarcode][1] += len(list_edited_conv)
        dict_sOutFreq[sBarcode][2] += len(list_other_conv)
        dict_sOutRead[sBarcode][0] += list_WT_conv
        dict_sOutRead[sBarcode][1] += list_edited_conv
        dict_sOutRead[sBarcode][2] += list_other_conv

        list_WT_opti      = [sReadID for sReadID in list_sColumn[4].split(',') if sReadID]
        list_edited_opti  = [sReadID for sReadID in list_sColumn[5].split(',') if sReadID]
        list_other_opti   = [sReadID for sReadID in list_sColumn[6].split(',') if sReadID]

        dict_sOutFreq[sBarcode][3] += len(list_WT_opti)
        dict_sOutFreq[sBarcode][4] += len(list_edited_opti)
        dict_sOutFreq[sBarcode][5] += len(list_other_opti)
        dict_sOutRead[sBarcode][3] += list_WT_opti
        dict_sOutRead[sBarcode][4] += list_edited_opti
        dict_sOutRead[sBarcode][5] += list_other_opti

        list_WT_none      = [sReadID for sReadID in list_sColumn[7].split(',') if sReadID]
        list_edited_none  = [sReadID for sReadID in list_sColumn[8].split(',') if sReadID]
        list_other_none   = [sReadID for sReadID in list_sColumn[9].split(',') if sReadID]

        dict_sOutFreq[sBarcode][6] += len(list_WT_none)
        dict_sOutFreq[sBarcode][7] += len(list_edited_none)
        dict_sOutFreq[sBarcode][8] += len(list_other_none)
        dict_sOutRead[sBarcode][6] += list_WT_none
        dict_sOutRead[sBarcode][7] += list_edited_none
        dict_sOutRead[sBarcode][8] += list_other_none

    #loop END: sReadLine
    InFile.close()
#def END: load_read_data_vOfftarget3


def temp_output_in_parts_vOfftarget2_Intended (sTempDir, dict_cPE, list_sSplitFiles, sSample, nNoSplits):

    nFileCnt   = len(list_sSplitFiles)
    list_nBins = [[int(nFileCnt * (i + 0) / nNoSplits), int(nFileCnt * (i + 1) / nNoSplits)] for i in range(nNoSplits)]

    for nStart, nEnd in list_nBins:
        list_sSubSplit   = list_sSplitFiles[nStart:nEnd]
                            #intendded, mismatch, and no scafford (WT, ALT, Others for each)
        dict_sOutFreq_H  = {sBarcode: [0, 0, 0, 0, 0] for sBarcode in dict_cPE}
        dict_sOutRead_H  = {sBarcode: [[], [], [], [], []] for sBarcode in dict_cPE}

        for i, sSplitFile in enumerate(list_sSubSplit):
            print('Output Temp Pickle %s/%s -- %s' % ((i+nStart), nEnd, sSplitFile))
            sSplitTag  = '_'.join(sSplitFile.split('.')[1].split('_')[-2:])
            sTempFile  = '%s/%s/%s'         % (sTempDir, sSample, sSplitTag)
            sInFile    = '%s/%s.reads.txt'  % (sTempFile, sSplitTag)
            print(sInFile)

            assert os.path.isfile(sInFile)
            load_read_data_vOfftarget2_Intended (dict_sOutFreq_H, dict_sOutRead_H, sInFile)
        #loop END: sSplitFile

        sOutFile  = '%s/%s.TempFreq%s-%s.data'    % (sTempDir, sSample, nStart, nEnd)
        OutFile  = open(sOutFile, 'wb')
        pickle.dump(dict_sOutFreq_H, OutFile)
        OutFile.close()

        sOutFile  = '%s/%s.TempReads%s-%s.data'    % (sTempDir, sSample, nStart, nEnd)
        OutFile  = open(sOutFile, 'wb')
        pickle.dump(dict_sOutRead_H, OutFile)
        OutFile.close()
    #loop END: nStart, nEnd
#def END: temp_output_in_parts


def load_read_data_vOfftarget2_Intended (dict_sOutFreq, dict_sOutRead, sInFile):

    InFile       = open(sInFile, 'r')
    for sReadLine in InFile:

        list_sColumn  = sReadLine.strip('\n').split('\t')
        sBarcode      = list_sColumn[0]
        list_WT       = [sReadID for sReadID in list_sColumn[1].split(',') if sReadID]
        list_edited   = [sReadID for sReadID in list_sColumn[2].split(',') if sReadID]
        list_intended = [sReadID for sReadID in list_sColumn[3].split(',') if sReadID]
        list_mismatch = [sReadID for sReadID in list_sColumn[4].split(',') if sReadID]
        list_other    = [sReadID for sReadID in list_sColumn[5].split(',') if sReadID]

        dict_sOutFreq[sBarcode][0] += len(list_WT)
        dict_sOutFreq[sBarcode][1] += len(list_edited)
        dict_sOutFreq[sBarcode][2] += len(list_intended)
        dict_sOutFreq[sBarcode][3] += len(list_mismatch)
        dict_sOutFreq[sBarcode][4] += len(list_other)

        dict_sOutRead[sBarcode][0] += list_WT
        dict_sOutRead[sBarcode][1] += list_edited
        dict_sOutRead[sBarcode][2] += list_intended
        dict_sOutRead[sBarcode][3] += list_mismatch
        dict_sOutRead[sBarcode][4] += list_other

    #loop END: sReadLine
    InFile.close()
#def END: load_read_data_vOfftarget2_Intended


def gzip_fastq (sInDir, sFastqTag, bTestRun):
    sScript = 'pigz --fast -c %s/%s.fastq > %s/%s.fastq.gz' % (sInDir, sFastqTag, sInDir, sFastqTag)
    if bTestRun: print(sScript)
    else:        os.system(sScript)
#def END: gzip_fastq


def gzip_fastq_list (sInDir, sFastqTag, bTestRun):

    sWorkDir      = '%s/split' % sInDir
    list_sFQFiles = os.listdir(sWorkDir)

    for sFQFile in list_sFQFiles:

        if not sFQFile.endswith('.fq'):continue
        sScript = 'pigz --fast -c %s/%s > %s/%s.gz' % (sWorkDir, sFQFile, sWorkDir, sFQFile)

        if bTestRun: print(sScript)
        else:        os.system(sScript)
    #loop END: sFQFile
#def END: gzip_fastq


def extract_reads_by_readID (sSample, sInDir, sFastqTag, sOutDir, nCores, nTop):

    sInFastq       = '%s/%s.fastq.gz'       % (sInDir, sFastqTag)
    sOthersDir     = '%s/others'            % sOutDir
    sInFile        = '%s/%s.readIDs.txt'    % (sOthersDir, sSample)
    InFile         = open(sInFile, 'r')
    dict_sReadData = {}

    for sReadLine in InFile:

        list_sColumn = sReadLine.strip('\n').split('\t')

        sBarcode     = list_sColumn[0]
        list_sReadID = list_sColumn[1].split(',')

        if sBarcode not in dict_sReadData:
            dict_sReadData[sBarcode] = ''
        dict_sReadData[sBarcode] = list_sReadID
    #loop END: sReadLine
    InFile.close()

    list_sBasicStats  = [[sBarcode, len(dict_sReadData[sBarcode])] for sBarcode in dict_sReadData]
    list_sBasicStats  = sorted(list_sBasicStats, key=lambda e:e[1], reverse=True)

    list_sTopBarcodes = list_sBasicStats[:nTop]
    sTopOutDir        = '%s/%s_top%s' % (sOthersDir, sSample, nTop)
    os.makedirs(sTopOutDir, exist_ok=True)

    sOutFile_toplist  = '%s/%s_top%s.txt' % (sOthersDir, sSample, nTop)
    OutFile           = open(sOutFile_toplist, 'w')
    list_sParameters  = []
    for sBarcode, nReadCnt in list_sTopBarcodes:

        sOut = '%s\t%s\n' % (sBarcode, nReadCnt)
        OutFile.write(sOut)

        sReadIDFile = '%s/%s/%s.readIDs.txt'   % (sOthersDir, sSample, sBarcode)
        sOutFile    = '%s/%s.seqs.txt'         % (sTopOutDir, sBarcode)
        list_sParameters.append([sReadIDFile, sInFastq, sOutFile])
    #loop END: sBarcode, nReadCnt
    OutFile.close()

    p = mp.Pool(nCores)
    p.map_async(run_seqkit_grep, list_sParameters).get()
#def END: extract_reads_by_readID

def extract_reads_by_readID_for_CRISPResso (nCores, sSample, sInDir, sOutDir, sFastqTag, sBarcodeFile, sTopBarcodeFile, list_sSplitFile, sError):

    nBins               = 4
    sInFastq            = '%s/%s.fastq.gz'    % (sInDir, sFastqTag)
    dict_cPE            = load_PE_input_v2(sBarcodeFile)
    list_sTopBarcode    = [sReadLine.strip('\n') for sReadLine in open(sTopBarcodeFile)]
    nTotalCnt           = len(list_sTopBarcode)

    dict_sRunBins       = {}
    list_nIndexKey      = [[i + 1, int(nTotalCnt * (i) / nBins), int(nTotalCnt * (i + 1) / nBins)] for i in
                            range(nBins)]
    for nIndex, nStart, nEnd in list_nIndexKey:
        if nIndex not in dict_sRunBins:
            dict_sRunBins[nIndex] = ''
        dict_sRunBins[nIndex] = list_sTopBarcode[nStart:nEnd]
    #loop END: nIndex, nStart, nEnd
    #1-node06 2-node01 3-node02 4-node03
    nRunNo            = 2
    list_sRunBarcodes = dict_sRunBins[nRunNo]

    print('list_sRunBarcodes', len(list_sRunBarcodes))

    dict_sBarcodeFiles  = get_files_for_seqtk (sOutDir, sError, sSample, list_sSplitFile, list_sRunBarcodes)

    list_sParameters1   = []
    list_sParameters2   = []

    for i, sBarcode in enumerate(list_sRunBarcodes):

        list_sInFiles = dict_sBarcodeFiles[sBarcode]
        sTempOutDir   = '%s/%s/temp/%s/forSeqKit'   % (sOutDir, sError, sSample)
        os.makedirs(sTempOutDir, exist_ok=True)
        sReadOut      = '%s/%s.readIDs.txt'         % (sTempOutDir, sBarcode)

        list_sParameters1.append([sBarcode, i, nTotalCnt, list_sInFiles, sReadOut])

        sFastqOutDir  = '%s/%s/temp/%s/fastq'   % (sOutDir, sError, sSample)
        os.makedirs(sFastqOutDir, exist_ok=True)

        sFastqTmp     = '%s/%s.fastq.tmp'  % (sFastqOutDir, sBarcode)
        sFastqOut     = '%s/%s.fastq'      % (sFastqOutDir, sBarcode)
        list_sParameters2.append([sReadOut, sInFastq, sFastqTmp, sFastqOut, nRunNo])
    #loop END: sBarcode, nReadCnt

    print('list_sParameters1', len(list_sParameters1))
    print('list_sParameters2', len(list_sParameters2))

    p = mp.Pool(nCores)
    #p.map_async(make_files_for_seqtk, list_sParameters1).get()
    p.map_async(run_seqkit_grep, list_sParameters2).get()
#def END: extract_reads_by_readID_for_indelsearcher


def extract_reads_by_readID_for_CRISPResso_v2 (nCores, sSample, sInDir, sOutDir, sFastqTag, sBarcodeFile, sTopBarcodeFile, list_sSplitFile, sError, nBins):

    list_sTopBarcode    = [sReadLine.strip('\n') for sReadLine in open(sTopBarcodeFile)]
    nTotalCnt           = len(list_sTopBarcode)

    dict_sBarcodeFiles  = get_files_for_cat (sOutDir, sError, sSample, list_sSplitFile, list_sTopBarcode)
    print(len(dict_sBarcodeFiles))

    for i, sBarcode in enumerate(list_sTopBarcode):

        print('Cat %s | %s/%s' % (sBarcode, i+1, nTotalCnt))

        list_sInFiles   = dict_sBarcodeFiles[sBarcode]
        sCombinedOutDir = '%s/%s/temp/%s/combined'   % (sOutDir, sError, sSample)
        os.makedirs(sCombinedOutDir, exist_ok=True)

        sFastqOut       = '%s/%s.fastq'  % (sCombinedOutDir, sBarcode)
        sScript         = 'cat %s > %s'  % (' '.join(list_sInFiles), sFastqOut)

        sp.call(sScript, shell=True)
    #loop END: sBarcode, nReadCnt
#def END: extract_reads_by_readID_for_indelsearcher


def extract_reads_by_readID_for_CRISPResso_v3 (nCores, sSample, sInDir, sOutDir, sFastqTag, sBarcodeFile, sTopBarcodeFile, list_sSplitFile, sError, nBins):

    sCombinedOutDir     = '%s/%s/temp/%s/combined'      % (sOutDir, sError, sSample)
    sCombinedTempDir    = '%s/%s/temp/%s/combined_temp' % (sOutDir, sError, sSample)
    os.makedirs(sCombinedOutDir, exist_ok=True)
    os.makedirs(sCombinedTempDir, exist_ok=True)

    list_sExists        = [sFile.replace('.fastq', '') for sFile in os.listdir(sCombinedOutDir)]
    list_sTopBarcode    = [sReadLine.strip('\n') for sReadLine in open(sTopBarcodeFile)]

    list_sTopBarcode    = list(set(list_sTopBarcode) - set(list_sExists))
    dict_sBarcodeFiles  = get_files_for_cat (sOutDir, sCombinedOutDir, sError, sSample, list_sSplitFile, list_sTopBarcode)
    nTotalCnt           = len(list_sTopBarcode)

    for i, sBarcode in enumerate(list_sTopBarcode):

        list_sInFiles   = dict_sBarcodeFiles[sBarcode]

        print('Cat %s | %s/%s | Files:' % (sBarcode, i+1, nTotalCnt), len(list_sInFiles))

        sFastqOut       = '%s/%s.fastq'  % (sCombinedOutDir, sBarcode)
        sScript         = 'cat %s > %s'  % (' '.join(list_sInFiles), sFastqOut)
        sp.call(sScript, shell=True)

        '''
        nBins            = 50
        nTotalJobs       = len(list_sInFiles)
        list_nIndexKey   = [[i + 1, int(nTotalJobs * (i) / nBi
        ns), int(nTotalJobs * (i + 1) / nBins)] for i in
                          range(nBins)]
        list_sPartials   = []
        list_sParameters = []
        for nIndex, nFileS, nFileE in list_nIndexKey:
            list_sFiles = list_sInFiles[nFileS:nFileE]
            sOutFile    = '%s/%s.%s.fastq' % (sCombinedTempDir, sBarcode, nIndex)

            list_sParameters.append([list_sFiles, sOutFile])

            sp.call('fcat %s >  %s' % (' '.join(list_sFiles), sOutFile), shell=True)

            list_sPartials.append(sOutFile)
        #loop END: nIndex, nFileS, nFileE

        p = mp.Pool(nCores)
        p.map_async(run_fcat_cmd, list_sParameters).get()
        
        list_sReadLines   = []
        for sInFile in list_sInFiles:
            InFile = open(sInFile, 'r')
            list_sReadLines += InFile.readlines()
            InFile.close()
        #loop END: sInFile

        OutFile = open(sFastqOut, 'w')
        for sReadLine in list_sReadLines:
            OutFile.write(sReadLine)
        #loop END: sReadLine
        OutFile.close()
        '''
    #loop END: sBarcode, nReadCnt
#def END: extract_reads_by_readID_for_indelsearcher


def get_files_for_cat (sOutDir, sCombinedOutDir, sError, sSample, list_sSplitFile, list_sBarcode):

    dict_sBarcode = {sBarcode: [] for sBarcode in list_sBarcode}

    for sSplitFile in list_sSplitFile:
        sSplitTag     = '_'.join(sSplitFile.split('.')[1].split('_')[-2:])
        sTempInDir    = '%s/%s/temp/%s/%s'        % (sOutDir, sError, sSample, sSplitTag)
        list_sInFile  = os.listdir(sTempInDir)
        nCnt = 0
        for sFile in list_sInFile:
            sBarcode      = sFile.split('.')[0]
            sInFile       = '%s/%s'       % (sTempInDir, sFile)
            sCheckOutFile = '%s/%s.fastq' % (sCombinedOutDir, sBarcode)

            try:
                dict_sBarcode[sBarcode].append(sInFile)
                nCnt += 1
            except KeyError: continue
            #try END:
        #loop END: sFile

        print(sSplitTag, nCnt)
    #loop END: sSplitFile
    return dict_sBarcode
#def END: get_files_for_cat


def run_fcat_cmd (list_sParameter):
    list_sFiles = list_sParameter[0]
    sOutFile    = list_sParameter[1]
    print(sOutFile)
    sp.call('fcat %s >  %s' % (' '.join(list_sFiles), sOutFile), shell=True)
#def END: run_fastq_cp_cmd


def make_files_for_seqtk (list_sParameter):

    sBarcode     = list_sParameter[0]
    i            = list_sParameter[1]
    nTotal       = list_sParameter[2]
    list_sInFile = list_sParameter[3]
    sReadOut     = list_sParameter[4]

    print('Processing %s : %s / %s' % (sBarcode, i + 1, nTotal))

    list_sReadIDs = []
    for sFile in list_sInFile:
        sReadLine = open(sFile).readline().strip('\n').split('\t')
        sBarcode = sReadLine[0]
        list_sReadIDs += sReadLine[1].split(',')
    # loop END: sFile

    # list_sReadIDs = ['@%s' % sReadID for sReadID in list_sReadIDs]

    OutFile = open(sReadOut, 'w')
    sOut = ''.join(['%s\n' % sReadID for sReadID in list_sReadIDs])
    OutFile.write(sOut)
    OutFile.close()
#def END: make_files_for_seqtk


def combined_output_fastq (sSample, sInDir, sFastqTag, sOutDir, nCores, nTop):

    sInFastq       = '%s/%s.fastq.gz'       % (sInDir, sFastqTag)
    sOthersDir     = '%s/others'            % sOutDir
    sInFile        = '%s/%s.readIDs.txt'    % (sOthersDir, sSample)
    InFile         = open(sInFile, 'r')
    dict_sReadData = {}
    for sReadLine in InFile:

        list_sColumn = sReadLine.strip('\n').split('\t')

        sBarcode     = list_sColumn[0]
        list_sReadID = list_sColumn[1].split(',')

        if sBarcode not in dict_sReadData:
            dict_sReadData[sBarcode] = ''
        dict_sReadData[sBarcode] = list_sReadID
    #loop END: sReadLine
    InFile.close()

    list_sBasicStats  = [[sBarcode, len(dict_sReadData[sBarcode])] for sBarcode in dict_sReadData]
    list_sBasicStats  = sorted(list_sBasicStats, key=lambda e:e[1], reverse=True)

    sIndelTemp = '%s/%s_forIndel'       % (sOthersDir, sSample)
    os.makedirs(sIndelTemp, exist_ok=True)

    list_sParameters  = []
    for sBarcode, nReadCnt in list_sBasicStats:

        sOut = '%s\t%s\n' % (sBarcode, nReadCnt)
        sReadIDFile = '%s/%s/%s.readIDs.txt'   % (sOthersDir, sSample, sBarcode)
        sOutFile    = '%s/%s.seqs.txt'         % (sIndelTemp, sBarcode)
        list_sParameters.append([sReadIDFile, sInFastq, sOutFile])
    #loop END: sBarcode, nReadCnt

    sIndelOut  = '%s/forIndelSearcher/%s-PE'  % (sOutDir, sSample)
    os.makedirs(sIndelOut, exist_ok=True)

    sOutFastq = '%s/%s.others.fastq' % (sIndelOut, sSample)
    ## Partially Cat files into Final File ##
    nBins       = 100
    list_sFiles = [sData[2] for sData in list_sParameters]
    nTotalJobs  = len(list_sFiles)
    list_nIndexKey = [[i + 1, int(nTotalJobs * (i) / nBins), int(nTotalJobs * (i + 1) / nBins)] for i in
                      range(nBins)]
    for nIndex, nFileS, nFileE in list_nIndexKey:
        list_sInFiles = list_sFiles[nFileS:nFileE]
        sOutFile      = '%s/%s.part%s.fastq' % (sIndelOut, sSample, nIndex)
        print(sOutFile)

        sp.call('cat %s >  %s' % (' '.join(list_sInFiles), sOutFile), shell=True)
    #loop END: nIndex, nFileS, nFileE
    sCmd      = 'cat %s/*part*.fastq > %s'    % (sIndelOut, sOutFastq)
    sp.call(sCmd, shell=True)

    print('%s-PE\t%s.others' % (sSample, sSample))
#def END: combined_output_fastq


def mp_combined_output_fastq_v2 (nCores, sSample, sOutputDir, sBarcodeFile, list_sSplits, sError):

    sWorkDir       = '%s/%s' % (sOutputDir, sError)

    dict_cPE       = load_PE_input_v2(sBarcodeFile)
    list_sBarcodes = list(dict_cPE.keys())
    print('list_sBarcodes', len(list_sBarcodes))
    ## Organize barcode texts to same folder ##
    sJustNGSDir = '%s/JustNGS' % sWorkDir

    list_sParameters = []
    for sBarcode in list_sBarcodes[:40]:
        sCombineDir = '%s/%s' % (sJustNGSDir, sBarcode)
        os.makedirs(sCombineDir, exist_ok=True)
        list_sParameters.append([sWorkDir, sSample, sBarcode, sCombineDir, list_sSplits])
    #loop END: sBarcode

    print('list_sParameters', len(list_sParameters))

    p = mp.Pool(nCores)
    p.map_async(run_fastq_cp_cmd, list_sParameters).get()
#def END: combined_output_fastq_v2


def run_fastq_cp_cmd (list_sParameters):

    sWorkDir     = list_sParameters[0]
    sSample      = list_sParameters[1]
    sBarcode     = list_sParameters[2]
    sCombineDir  = list_sParameters[3]
    list_sSplits = list_sParameters[4]

    print('Copying %s' % sBarcode)

    for sSplitFile in list_sSplits:
        sSplitTag     = '_'.join(sSplitFile.split('.')[1].split('_')[-2:])
        sTempDir      = '%s/temp/%s/%s' % (sWorkDir, sSample, sSplitTag)
        sBarcodeDir   = '%s/bybarcodes' % sTempDir

        sBarcodeFile1 = '%s/%s.txt'    % (sBarcodeDir, sBarcode)
        sBarcodeFile2 = '%s/%s.%s.txt' % (sCombineDir, sBarcode, sSplitTag)

        if not os.path.isfile(sBarcodeFile1): continue

        sCmd = 'cp %s %s' % (sBarcodeFile1, sBarcodeFile2)
        os.system(sCmd)
    #loop END: sSplitFile

    print('Copying %s ------- DONE' % sBarcode)
#def END: run_fastq_cp_cmd


def extract_reads_by_readID_for_indelsearcher (sSample, sInDir, sFastqTag, sOutDir, nCores, nTop):

    sInFastq       = '%s/%s.fastq.gz'       % (sInDir, sFastqTag)
    sOthersDir     = '%s/others'            % sOutDir
    sInFile        = '%s/%s.readIDs.txt'    % (sOthersDir, sSample)
    InFile         = open(sInFile, 'r')
    dict_sReadData = {}
    for sReadLine in InFile:

        list_sColumn = sReadLine.strip('\n').split('\t')

        sBarcode     = list_sColumn[0]
        list_sReadID = list_sColumn[1].split(',')

        if sBarcode not in dict_sReadData:
            dict_sReadData[sBarcode] = ''
        dict_sReadData[sBarcode] = list_sReadID
    #loop END: sReadLine
    InFile.close()

    list_sBasicStats  = [[sBarcode, len(dict_sReadData[sBarcode])] for sBarcode in dict_sReadData]
    list_sBasicStats  = sorted(list_sBasicStats, key=lambda e:e[1], reverse=True)

    sIndelTemp = '%s/%s_forIndel'       % (sOthersDir, sSample)
    os.makedirs(sIndelTemp, exist_ok=True)

    sOutFile_fulllist = '%s/%s_full.txt' % (sOthersDir, sSample)
    OutFile           = open(sOutFile_fulllist, 'w')
    list_sParameters  = []
    for sBarcode, nReadCnt in list_sBasicStats:

        sOut = '%s\t%s\n' % (sBarcode, nReadCnt)
        OutFile.write(sOut)

        sReadIDFile = '%s/%s/%s.readIDs.txt'   % (sOthersDir, sSample, sBarcode)
        sOutFile    = '%s/%s.seqs.txt'         % (sIndelTemp, sBarcode)
        list_sParameters.append([sReadIDFile, sInFastq, sOutFile])
    #loop END: sBarcode, nReadCnt
    OutFile.close()

    p = mp.Pool(nCores)
    p.map_async(run_seqkit_grep, list_sParameters).get()

    sIndelOut  = '%s/forIndelSearcher/%s-PE'  % (sOutDir, sSample)
    os.makedirs(sIndelOut, exist_ok=True)

    sOutFastq = '%s/%s.others.fastq' % (sIndelOut, sSample)
    ## Partially Cat files into Final File ##
    nBins       = 100
    list_sFiles = [sData[2] for sData in list_sParameters]
    nTotalJobs  = len(list_sFiles)
    list_nIndexKey = [[i + 1, int(nTotalJobs * (i) / nBins), int(nTotalJobs * (i + 1) / nBins)] for i in
                      range(nBins)]
    for nIndex, nFileS, nFileE in list_nIndexKey:
        list_sInFiles = list_sFiles[nFileS:nFileE]
        sOutFile      = '%s/%s.part%s.fastq' % (sIndelOut, sSample, nIndex)
        print(sOutFile)

        sp.call('cat %s >  %s' % (' '.join(list_sInFiles), sOutFile), shell=True)
    #loop END: nIndex, nFileS, nFileE
    sCmd      = 'cat %s/*part*.fastq > %s'    % (sIndelOut, sOutFastq)
    sp.call(sCmd, shell=True)

    print('%s-PE\t%s.others' % (sSample, sSample))
#def END: extract_reads_by_readID_for_indelsearcher


def mp_run_crispresso (nCores, sAnalysis, sSample, sOutputDir, sBarcodeFile, list_sSplitFile):

    sUser      = 'GS'
    sWorkDir   = '/extdata1/CRISPR_toolkit/Indel_searcher_2'
    sInputDir  = '%s/Input/%s'     % (sWorkDir, sUser)
    sOutputDir = '%s/Output/%s'    % (sWorkDir, sUser)
    sUserDir   = '%s/User/%s'      % (sWorkDir, sUser)

    dict_cPE       = load_PE_input_v2(sBarcodeFile)
    make_reference_crispresso (dict_cPE, sInputDir, sAnalysis, sSample)
    pass
#def END: mp_run_crispresso


def make_reference_crispresso(dict_cPE, sInputDir, sAnalysis, sSample):

    sRefDir        = '%s/Reference/%s/%s' % (sInputDir, sAnalysis, sSample)
    os.makedirs(sRefDir, exist_ok=True)

    list_sBarcodes = list(dict_cPE.keys())

    nBins           = 5
    nTotalJobs      = len(list_sBarcodes)
    list_nIndexKey  = [[i + 1, int(nTotalJobs * i / nBins), int(nTotalJobs * (i + 1) / nBins)] for i in
                        range(nBins)]
    for nIndex, nStart, nEnd in list_nIndexKey:

        list_sSubset = list_sBarcodes[nStart:nEnd]
        sOutFile     = '%s/Reference_part%s.fa' % (sRefDir, nIndex)
        OutFile      = open(sOutFile, 'w')

        for sBarcode in list_sSubset:
            cPE  = dict_cPE[sBarcode]
            sOut = '>%s:%s\n%s\n' % (sBarcode, cPE.sWTSeq, cPE.sRefSeq)
            OutFile.write(sOut)
        #loop END: sBarcode
        OutFile.close()
    #loop END: nIndex
#def END: make_reference_Crispresso


def mp_combined_output_fastq_v3 (nCores, sSample, sOutputDir, sBarcodeFile, list_sSplits, sError):

    sWorkDir       = '%s/%s' % (sOutputDir, sError)
    dict_cPE       = load_PE_input_v2(sBarcodeFile)
    list_sBarcodes = list(dict_cPE.keys())

    sOutDir         = '%s/JustNGS' % sWorkDir
    list_sRun       = []
    for sBarcode in list_sBarcodes:
        sOutFile = '%s/%s.txt'  % (sOutDir, sBarcode)

        if not os.path.isfile(sOutFile): list_sRun.append(sBarcode)  # if Barcode done, do not rerun
        else: continue
    #loop END: sBarcode
    print('Total list_sBarcodes', len(list_sBarcodes))
    print('ToRun list_sBarcodes', len(list_sRun))


    nBins           = nCores
    nTotalJobs      = len(list_sRun)
    list_nIndexKey  = [[i + 1, int(nTotalJobs * i / nBins), int(nTotalJobs * (i + 1) / nBins)] for i in
                        range(nBins)]

    list_sParameters = []
    for nIndex, nStart, nEnd in list_nIndexKey:

        list_sSubset = list_sRun[nStart:nEnd]

        list_sParameters.append([sWorkDir, sSample, list_sSplits, list_sSubset])
    #loop END: nIndex, nStart, nEnd

    combined_output_fastq_v3([sWorkDir, sSample, list_sSplits, list_sBarcodes])

    #p = mp.Pool(nCores)
    #p.map_async(combined_output_fastq_v3, list_sParameters).get()
#def END: run_fastq_cp_cmd


def combined_output_fastq_v3 (list_sParmeter):
    sWorkDir        = list_sParmeter[0]
    sSample         = list_sParmeter[1]
    list_sSplits    = list_sParmeter[2]
    list_sBarcodes  = list_sParmeter[3]

    for sBarcode in list_sBarcodes:

        print('Combining %s' % sBarcode)

        list_sFiles1 = []
        for sSplitFile in list_sSplits:
            sSplitTag = '_'.join(sSplitFile.split('.')[1].split('_')[-2:])

            sInDir    = '%s/temp/%s/%s/bybarcodes' % (sWorkDir, sSample, sSplitTag)
            sInFile   = '%s/%s.txt'                % (sInDir, sBarcode)

            if os.path.isfile(sInFile): list_sFiles1.append(sInFile)
            else:                             continue
        #loop END: sSplitFile

        if not list_sFiles1: continue ## Barcode not in data.

        nTotalJobs     = len(list_sFiles1)

        if   nTotalJobs == 1:            nBins = 1
        elif nTotalJobs <= 100:          nBins = 5
        elif 100  < nTotalJobs <= 500:   nBins = 20
        elif 500  < nTotalJobs <= 1000:  nBins = 50
        elif 1000 < nTotalJobs <= 3000:  nBins = 80
        else:                            nBins = 100

        if nBins > nTotalJobs: nBins = nTotalJobs

        list_nIndexKey = [[i + 1, int(nTotalJobs * i / nBins), int(nTotalJobs * (i + 1) / nBins)] for i in
                          range(nBins)]

        print('list_sFiles1', nTotalJobs, nBins)
        sOutDir        = '%s/temp/%s/combined_barcodes/%s' % (sWorkDir, sSample, sBarcode)
        os.makedirs(sOutDir, exist_ok=True)

        list_sFiles2  = []
        for nIndex, nFileS, nFileE in list_nIndexKey:

            list_sSubset = list_sFiles1[nFileS:nFileE]

            sOutFile  = '%s/%s.%s-%s.txt' % (sOutDir, sBarcode, nFileS, nFileE)
            list_sFiles2.append(sOutFile)

            sCmd      = 'cat %s > %s' % (' '.join(list_sSubset), sOutFile)
            #print('Summary_Part%s %s | Cmd Length %s' % (nIndex, len(list_sSubset), len(sCmd)))

            sp.call(sCmd, shell=True)
        #loop END: nIndex, nFileS, nFileE

        sOutDir2  = '%s/JustNGS' % sWorkDir
        os.makedirs(sOutDir2, exist_ok=True)
        sOutFile2 = '%s/%s.txt'  % (sOutDir2, sBarcode)
        sCmd      = 'cat %s > %s' % (' '.join(list_sFiles2), sOutFile2)
        os.system(sCmd)
        print('Combining %s ------- DONE' % sBarcode)
    #loop END: sBarcode
#def END: combined_output_fastq_v3


def combined_output_fastq_vCheckLoss (nCores, sSample, sOutputDir, sBarcodeFile, list_sSplits, sError):

    dict_cPE       = load_PE_input_v2(sBarcodeFile)
    list_sBarcodes = list(dict_cPE.keys())
    nTotalCnt      = len(list_sBarcodes)
    sOutDir        = '%s/%s'        % (sOutputDir, sError)
    sCheckLossDir  = '%s/checkloss' % sOutDir

    dict_sBarcode  = {sBarcode: [] for sBarcode in list_sBarcodes}

    for sSplitFile in list_sSplits:
        sSplitTag       = '_'.join(sSplitFile.split('.')[1].split('_')[-2:])
        sTempInDir      = '%s/%s' % (sCheckLossDir, sSplitTag)
        list_sInFile    = os.listdir(sTempInDir)
        nCnt = 0
        for sFile in list_sInFile:
            sBarcode = sFile.split('.')[0]
            sInFile = '%s/%s' % (sTempInDir, sFile)
            try:
                dict_sBarcode[sBarcode].append(sInFile)
                nCnt += 1
            except KeyError:
                continue
            #try END:
        #loop END: sFile
    #loop END: sSplitFile

    sCombinedOutDir  = '%s/checkloss_combined' % sOutDir
    os.makedirs(sCombinedOutDir, exist_ok=True)

    for i, sBarcode in enumerate(list_sBarcodes):
        list_sInFiles = dict_sBarcode[sBarcode]

        print('Cat %s | %s/%s | Files:' % (sBarcode, i + 1, nTotalCnt), len(list_sInFiles))

        sFastqOut   = '%s/%s.fastq' % (sCombinedOutDir, sBarcode)
        sScript     = 'cat %s > %s' % (' '.join(list_sInFiles), sFastqOut)
        sp.call(sScript, shell=True)
    #loop END: i, sBarcode

#def END: run_fastq_cp_cmd



def run_seqkit_grep (sParameters):
    sReadIDFile = sParameters[0]
    sInFastq    = sParameters[1]
    sTmpFile    = sParameters[2]
    sOutFile    = sParameters[3]
    sRunNo      = sParameters[4]
    sScript     = 'seqkit grep -f %s %s > %s;' % (sReadIDFile, sInFastq, sTmpFile)
    sScript    += 'mv %s %s' % (sTmpFile, sOutFile)
    print(sOutFile, sRunNo)
    sp.call(sScript, shell=True)
#def END: run_seqkit_grep


def analyze_top_barcodes (sSample, sBarcodeFile, sOutDir, nTop, nLineCnt):

    dict_cPE       = load_PE_input(sBarcodeFile)
    sOthersDir     = '%s/others'        % sOutDir
    sTopOutDir     = '%s/%s_top%s'      % (sOthersDir, sSample, nTop)
    sInFile        = '%s/%s_top%s.txt'  % (sOthersDir, sSample, nTop)
    InFile         = open(sInFile, 'r')
    list_nTop      = [sReadLine.strip('\n').split('\t') for sReadLine in InFile]
    InFile.close()
    dict_sOutput   = {}

    for sBarcode, nReadCnt in list_nTop:

        print(sBarcode, nReadCnt)

        cPE          = dict_cPE[sBarcode]
        nWTLen       = len(cPE.sWTSeq)
        sInFile      = '%s/%s.seqs.txt' % (sTopOutDir, sBarcode)
        InFile       = open(sInFile, 'r')
        sKey         = sBarcode

        if sKey not in dict_sOutput:
            dict_sOutput[sKey] = [0 for i in range(len(cPE.sWTSeq))]

        for i, sReadLine in enumerate(InFile):

            if i % 4 == 0: sReadID = sReadLine.replace('\n', '')
            if i % 4 != 1: continue

            sNGSSeq       = sReadLine.replace('\n', '').upper()
            list_sReFound = []

            for sReIndex in regex.finditer(sBarcode, sNGSSeq, overlapped=True):
                nIndexStart = sReIndex.start()
                nIndexEnd   = sReIndex.end()

                list_sReFound.append([nIndexStart, nIndexEnd])
            #loop END: sReIndex
            nIndexS, nIndexE = list_sReFound[0] ## Assume single barcode per read
            sBarcodeCheck    = sNGSSeq[nIndexS:nIndexE]
            assert sBarcode == sBarcodeCheck
            sTargetSeq       = sNGSSeq[nIndexE:nIndexE+nWTLen]
            list_sMM = [' ' if sWT == sTar else '*' for sWT, sTar in zip(cPE.sWTSeq, sTargetSeq)]
            sMMSeq   = ''.join(list_sMM)

            for i, sMM in enumerate(sMMSeq):
                if sMM == ' ': continue
                else: dict_sOutput[sBarcode][i] += 1
            #loop END: i, sMM
        #loop END: i, sReadLine
        InFile.close()
    #loop END: cPE

    sOutFile = '%s/%s_top%s.tally.txt' % (sOthersDir, sSample, nTop)
    OutFile  = open(sOutFile, 'w')

    for sBarcode, nReadCnt in list_nTop:
        list_nNorms = [(nCnt/nLineCnt)*1000000 for nCnt in dict_sOutput[sBarcode]]
        sOut        = '%s\t%s\t%s\n' % (sBarcode, nReadCnt, ','.join([str(nCnt) for nCnt in list_nNorms]))
        OutFile.write(sOut)
    #loop END: sBarcode, nReadCnt
    OutFile.close()
#def END: analyze_top_barcodes


def pbs_sort_by_barcode (sAnalysis, sBarcodeFile, sFastqFile, sOutDir,  sQueue, bTestRun, sTimeLimit):

    nBins            = 50
    list_cData       = load_PE_input (sBarcodeFile)
    sTmpSRC          = copy_temp_core_script ()
    nTotalCnt        = len(list_cData)
    list_nBin        = [[int(nTotalCnt * (i + 0) / nBins), int(nTotalCnt * (i + 1) / nBins)] for i in range(nBins)]
    list_sParameters = []

    sTempDir         = '%s/temp/%s' % (sOutDir, sAnalysis)
    os.makedirs(sTempDir, exist_ok=True)

    for nBinS, nBinE in list_nBin:
        sOutFreq        = '%s/tempout.%s-%s.txt'   % (sTempDir, nBinS, nBinE)
        sOutReadID      = '%s/readIDout.%s-%s.txt' % (sTempDir, nBinS, nBinE)
        list_sParameters.append([nBinS, nBinE, sOutFreq, sOutReadID])
    #loop END: nBinS, nBinE

    list_sHostnames = ['node01','node02','node03','node06']

    for i, sParameter in enumerate(list_sParameters):

        nBinS, nBinE, sOutFreq, sOutReadID = sParameter

        sHostName   = random.sample(list_sHostnames, 1)[0]
        sRunTag     = '%s.%s-%s'            % (sAnalysis, nBinS, nBinE)
        sJobName    = 'PJM.SortBar.%s'      % sRunTag
        sLogDir     = make_log_dir ()
        sTmpDir     = make_tmp_dir ()
        sLogFile    = '%s/%s.log'           % (sLogDir, sRunTag)
        sLogFile2   = '%s/%s.error.log'     % (sLogDir, sRunTag)
        sPBSScript  = '%s/%s.script.pbs'    % (sTmpDir, sJobName)

        sScript     = '#!/bin/bash\n'
        sScript    += '#PBS -q %s\n'             % sQueue
        sScript    += '#PBS -N %s\n'             % sJobName
        sScript    += '#PBS -o %s\n'             % sLogFile
        sScript    += '#PBS -e %s\n'             % sLogFile2
        sScript    += '#PBS -V\n'

        sScript    += '%s pbs_check_barcode '    % sTmpSRC
        sScript    += '%s '                      % nBinS
        sScript    += '%s '                      % nBinE
        sScript    += '%s '                      % sFastqFile
        sScript    += '%s '                      % sBarcodeFile
        sScript    += '%s '                      % sOutFreq
        sScript    += '%s '                      % sOutReadID

        OutFile = open(sPBSScript, 'w')
        OutFile.write(sScript)
        OutFile.close()

        sCmd    = 'qsub -l nodes=%s %s' % (sHostName, sPBSScript)

        if bTestRun: print(sScript);print(sCmd)
        else:        os.system(sCmd)
    #loop END: sParameter
#def END: pbs_sort_by_barcode


def pbs_sort_by_barcode_v2 (sAnalysis, sBarcodeFile, sFastqFile, sOutDir,  sQueue, bTestRun, sTimeLimit):

    nBins            = 50
    list_cData       = load_PE_input (sBarcodeFile)
    sTmpSRC          = copy_temp_core_script ()
    nTotalCnt        = len(list_cData)
    list_nBin        = [[int(nTotalCnt * (i + 0) / nBins), int(nTotalCnt * (i + 1) / nBins)] for i in range(nBins)]
    list_sParameters = []

    sTempDir         = '%s/temp/%s' % (sOutDir, sAnalysis)
    os.makedirs(sTempDir, exist_ok=True)

    for nBinS, nBinE in list_nBin:
        sOutFreq        = '%s/tempout.%s-%s.txt'   % (sTempDir, nBinS, nBinE)
        sOutReadID      = '%s/readIDout.%s-%s.txt' % (sTempDir, nBinS, nBinE)
        list_sParameters.append([nBinS, nBinE, sOutFreq, sOutReadID])
    #loop END: nBinS, nBinE

    list_sHostnames = ['node01','node02','node03','node06']

    for i, sParameter in enumerate(list_sParameters):

        nBinS, nBinE, sOutFreq, sOutReadID = sParameter

        sHostName   = random.sample(list_sHostnames, 1)[0]
        sRunTag     = '%s.%s-%s'            % (sAnalysis, nBinS, nBinE)
        sJobName    = 'PJM.SortBar.%s'      % sRunTag
        sLogDir     = make_log_dir ()
        sTmpDir     = make_tmp_dir ()
        sLogFile    = '%s/%s.log'           % (sLogDir, sRunTag)
        sLogFile2   = '%s/%s.error.log'     % (sLogDir, sRunTag)
        sPBSScript  = '%s/%s.script.pbs'    % (sTmpDir, sJobName)

        sScript     = '#!/bin/bash\n'
        sScript    += '#PBS -q %s\n'             % sQueue
        sScript    += '#PBS -N %s\n'             % sJobName
        sScript    += '#PBS -o %s\n'             % sLogFile
        sScript    += '#PBS -e %s\n'             % sLogFile2
        sScript    += '#PBS -V\n'

        sScript    += '%s pbs_check_barcode '    % sTmpSRC
        sScript    += '%s '                      % nBinS
        sScript    += '%s '                      % nBinE
        sScript    += '%s '                      % sFastqFile
        sScript    += '%s '                      % sBarcodeFile
        sScript    += '%s '                      % sOutFreq
        sScript    += '%s '                      % sOutReadID

        OutFile = open(sPBSScript, 'w')
        OutFile.write(sScript)
        OutFile.close()

        sCmd    = 'qsub -l nodes=%s %s' % (sHostName, sPBSScript)

        if bTestRun: print(sScript);print(sCmd)
        else:        os.system(sCmd)
    #loop END: sParameter
#def END: pbs_sort_by_barcode


def pbs_check_barcode (nBinS, nBinE, sInFastq, sBarcodeFile, sOutFreq, sOutReadID):

    nBinS          = int(nBinS)
    nBinE          = int(nBinE)
    list_cPE       = load_PE_input (sBarcodeFile)[nBinS:nBinE]
    dict_sOutput   = {}
    InFile         = open(sInFastq, 'r')
    for i, sReadLine in enumerate(InFile):

        if i % 4 == 0: sReadID = sReadLine.replace('\n', '')
        if i % 4 != 1: continue

        for n, cPE in enumerate(list_cPE):

            nWTSize = len(cPE.sWTSeq)
            nAltSize = len(cPE.sAltSeq)

            if cPE.sBarcode not in dict_sOutput:
                dict_sOutput[cPE.sBarcode] = {'WT': [], 'Alt': [], 'Other': []}

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
                break

            elif sAltSeqCheck == cPE.sAltSeq:
                dict_sOutput[cPE.sBarcode]['Alt'].append(sReadID)
                break

            elif sWTSeqCheck != cPE.sWTSeq and sAltSeqCheck != cPE.sAltSeq:
                dict_sOutput[cPE.sBarcode]['Other'].append(sReadID)
                break

            else: continue
        #loop END: cPE
    #loop END: i, sReadLine
    InFile.close()

    OutFile     = open(sOutFreq, 'w')
    OutFile2    = open(sOutReadID, 'w')
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


def get_guides_from_genome (sOutputDir, dict_sHGNC, dict_cRefGene, cGenome, dict_sTargetGenes):

    ## Flanking Sizes ##
    nUpFlank         = 24
    nDownFlank       = 3
    nBufferSize      = 100
    dict_sRE         = {'+': '[ACGT]GG',
                        '-': 'CC[ACGT]'}
    list_cFinal_Ref  = []
    list_sNotFound   = []
    for sGeneSym in dict_sTargetGenes:

        sHGNCID, sGroupName = dict_sTargetGenes[sGeneSym]

        if sGeneSym != 'CCN3': continue

        try: list_cRef = dict_cRefGene[sGeneSym] # Multiple Transcripts per GeneSym
        except KeyError:
            try:
                sGeneSym_hgnc = dict_sHGNC[sHGNCID]
                list_cRef = dict_cRefGene[sGeneSym_hgnc]

            except KeyError:
                list_sNotFound.append([sGeneSym, sHGNCID, sGroupName])
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
        cRef.sGroupName    = sGroupName
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

    sOutFile = '%s/20191111_PE_KOLib_Guides.txt'          % sOutputDir


    output_full_data (sOutFile, list_cFinal_Ref)

    sOutFile = '%s/20191111_PE_KOLib_Guides_NotFound.txt' % sOutputDir
    OutFile  = open(sOutFile, 'w')
    for sGeneSym, sHGNCID, sGroupName in list_sNotFound:
        sOut = '%s,%s,%s\n' % (sGeneSym, sHGNCID, sGroupName)
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
                    nSeqStart    = nIndexStart - nUpFlank
                    nSeqEnd      = nIndexEnd + nDownFlank
                    nGenomicS    = nExonS  + nSeqStart
                    nGenomicE    = nExonS  + nSeqEnd - 1
                    nCutSitePos  = nGenomicE - 9

                    if nGenomicS < nExonS or nGenomicE > nExonE:
                        sTarSeq = cGenome.fetch(cRef.sChrID, nGenomicS - 1, nGenomicE, sStrand).upper()
                    else: sTarSeq = sFullSeq[nSeqStart:nSeqEnd]

                    nBufferS     = nGenomicS - nBufferSize
                    nBufferE     = nGenomicE + nBufferSize
                    sBufferSeq   = cGenome.fetch(cRef.sChrID, nBufferS - 1, nBufferE, sStrand).upper()

                else:
                    nSeqStart    = nIndexStart - nDownFlank
                    nSeqEnd      = nIndexEnd + nUpFlank
                    nGenomicS    = nExonS  + nSeqStart
                    nGenomicE    = nExonS  + nSeqEnd - 1
                    nCutSitePos  = nGenomicS + 9 - 1

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
                sKey = 'gRNA-%s.%s-%s.%s' % (nGuideCnt, nSeqStart, nSeqEnd, sStrand)

                if sKey not in cRef.dict_sGuides[sExonKey]:
                    cRef.dict_sGuides[sExonKey][sKey] = ''
                cRef.dict_sGuides[sExonKey][sKey] = [sTarSeq, sBufferSeq, nCutSitePos, nCutPercent]
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
                sGuideSeq, sBufferSeq, nCutSitePos, nCutPercent = cRef.dict_sGuides[sExonKey][sGuideKey]

                sOut = '%s,%s,%s,%s,%s:%s-%s,%s,%s,%s,%s,%s,%s\n' \
                       % (cRef.sGeneSym, cRef.sNMID, cRef.sChrID, cRef.sStrand, cRef.nTxnStartPos, cRef.nTxnEndPos,
                          cRef.nTxnSize, sExonKey, sGuideKey, sGuideSeq, sBufferSeq, nCutSitePos,
                          nCutPercent)
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


def combined_analysis_output_freq (list_sAnalysis, list_sSamples, sBarcodeFile, sRun, ssError):

    sOutputDir    = '%s/output/%s_%s' % (sBASE_DIR, sAnalysis, sRun)
    sOutDir       = '%s/%s' % (sOutputDir, sError)
    os.makedirs(sOutDir, exist_ok=True)

    sTempDir            = '%s/temp'            % sOutDir
    sCombinedOutDir     = '%s/combined_output' % sOutDir
    os.makedirs(sCombinedOutDir, exist_ok=True)

    dict_cPE            = load_PE_input_v2(sBarcodeFile)
    list_sBarcodes      = list(dict_cPE.keys())
    nTotalJobs          = len(list_sBarcodes)
    print('dict_cPE', len(dict_cPE))
    print('list_sBarcodes', nTotalJobs)

#def END: combined_freq_output



def main():
    ## 10/31/2019 Prime Editing KO Screening ##

    #sAnalysis   = 'HKK_191230'
    #sAnalysis   = 'HKK_200413'
    #sAnalysis   = '200602_GSY'
    #sAnalysis   = 'HKK_200609'
    #sAnalysis   = 'HKK_201111' # PE 1st Round
    #sAnalysis   = 'HKK_201126' # PE Background
    #sAnalysis   = 'HKK_201214' # PE Full
    #sAnalysis   = 'HKK_210121' # PE Subpool 1-4
    #sAnalysis   = 'HKK_210316' # PE Offtarget Main
    #sAnalysis   = 'HKK_210319' # PE Offtarget Sub (Same NGS run as offtarget main)
    #sAnalysis   = 'HKK_210405'  #D4, D21
    #sAnalysis   = 'Nahye_210506'
    #sAnalysis   = 'Nahye_210507' #Background
    #sAnalysis   = 'MY_210513'
    #sAnalysis   = 'HKK_210906'  # PE OffTarget2 (2nd trial)
    #sAnalysis   = 'GSY_211008'  # PE OffTarget2 (2nd trial)
    #sAnalysis   = 'GSY_211008_HCT'  # PE OffTarget2 HCT files (2nd trial)
    #sAnalysis   = 'GSY_211012'   # PE OffTarget3
    #sAnalysis   = 'GSY_220303'   # PE OffTarget3
    #sAnalysis   = 'GSY_220324'    # PE OffTarget3
    #sAnalysis   = 'ENDO_221116'    # PE OffTarget3
    #sAnalysis   = 'ENDO_221116_PE3'    # PE OffTarget3
    #sAnalysis   = 'ENDO_221119_PE2_ADD'    # PE OffTarget3
    #sAnalysis   = 'ENDO_221119_PE3_ADD'    # PE OffTarget3
    #sAnalysis   = 'ENDO_221119_PE3_CTRL'    # PE OffTarget3

    #sAnalysis   = 'ENDO_221119_PE3_OFF'    # PE OffTarget3
    sAnalysis   = 'ENDO_221124_PE3_OFF'    # PE OffTarget3
    #sAnalysis   = 'ENDO_221116_BRCA'    # PE OffTarget3
    #sAnalysis   = 'ENDO_221126_BRCA'    # PE OffTarget3
    #sAnalysis   = 'ENDO_221126_PE3'    # PE OffTarget3
    #sAnalysis   = 'ENDO_221126_PE3_OFF'    # PE OffTarget3

    ## Multiplex Options ##
    sQueue      = 'workq'
    bTestRun    = False
    sTimeLimit  = '250:00:00'
    nCores      = 10
    nBins       = 10

    ## Main Input Dir ##
    sInputDir   = '%s/input'        % sBASE_DIR
    sOutputDir  = '%s/output'        % sBASE_DIR

    ## Rename SamplesIDs -> SampleName ##
    sDataDir    = '%s/%s'                  % (sInputDir, sAnalysis)
    sRenameList = '%s/RenameLists/%s.txt'  % (sInputDir, sAnalysis)
    #rename_samples (sDataDir, sRenameList)

    ## Run FLASH ##
    sFileList   = '%s/FileLists/%s.txt'  % (sInputDir, sAnalysis)
    dict_sFiles, doff = load_NGS_files (sAnalysis, sFileList)

    #run_FLASH (sAnalysis, sDataDir, dict_sFiles, doff, bTestRun)
    #sys.exit()

    ## HKK Modified Endo Barcode 2211 ###
    # Step 1: Preliminary #
    #sFileList = '%s/FileLists/%s_Step1.txt' % (sInputDir, sAnalysis)  # endo
    #dict_sFiles = load_NGS_files(sAnalysis, sFileList)
    #endo_sort_by_barcode_step1(sDataDir, dict_sFiles)
    # Step 2: Initalize ins-del-sub distribution   - Preliminary
    #sFileList   = '%s/FileLists/%s_Step2.txt' % (sInputDir, sAnalysis)  # endo
    #endo_sort_by_barcode_step2(sDataDir, dict_sFiles)
    ##############################################


    #endo_sort_by_barcode_readlevelcnt(sDataDir, dict_sFiles)
    #endo_sort_by_barcode_readlevelcnt_pe3 (sAnalysis, sDataDir, dict_sFiles)
    #endo_sort_by_barcode_readlevelcnt_ctrl(sAnalysis, sDataDir, dict_sFiles)
    endo_sort_by_barcode_readlevelcnt_pe3_OFF (sAnalysis, sDataDir, sOutputDir, dict_sFiles)

    #run_maund (sDataDir, dict_sFiles)

    ##############################

    sys.exit()

    dict_sRE    = {'ClinVar':  '[ACGT]{6}[T]{6}[ACGT]{24}',  'Profiling':   '[ACGT]{6}[T]{6}[ACGT]{24}',
                   'Subpool':  '[ACGT]{6}[T]{6}[ACGT]{24}',  'D4D21':       '[ACGT]{6}[T]{6}[ACGT]{24}',
                                                             'JustNGSReads':'[ACGT]{6}[T]{6}[ACGT]{23}',
                   'Offtarget':'[ACGT]{6}[T]{6}[ACGT]{23}',  'Offtarget2':  '[T]{6}[ACGT]{20}',
                   'Offtarget2-Intended': '[T]{6}[ACGT]{20}', 'Offtarget3': '[T]{6}[ACGT]{17}',
                   'Offtarget2-Intended-Test': '[T]{6}[ACGT]{20}', 'Endo_1': 'NULL'
                   }

    ## Error Type: ErrorProne, ErrorFree
    #list_sErrorType = ['ErrorFree', 'ErrorProne']
    list_sErrorType = ['ErrorFree']
    #list_sErrorType = ['ErrorProne']

    ## Guide Run List: Profiling, ClinVar
    #list_sGuideRun  = ['ClinVar', 'Profiling']
    #list_sGuideRun  = ['ClinVar']
    #list_sGuideRun  = ['Subpool']  # for HKK210121 and MY_210513
    #list_sGuideRun = ['Offtarget']  # for HKK_210316 | HKK_210319 | Nahye_210506 |
    #list_sGuideRun = ['Offtarget2'] # for HKK_210906 and GSY_211008 (2nd trial)
    #list_sGuideRun = ['Offtarget2-Intended'] # for HKK_210906 | GSY_211008 | GSY_211008_HCT (2nd trial)
    #list_sGuideRun = ['Offtarget2-Intended-Test'] # for HKK_210906 and GSY_211008 (2nd trial)
    #list_sGuideRun = ['Offtarget3'] # for GSY_211012, GSY_220303, GSY_220324 (Scaffold distinction)
    list_sGuideRun = ['Endo_1'] # ENDO_221116

    #list_sGuideRun  = ['D4D21'] # HKK_210405
    #list_sGuideRun  = ['JustNGSReads'] # HKK_201126 and HKK_210405

    for sRun in list_sGuideRun:

        sRE = dict_sRE[sRun]

        for sError in list_sErrorType:

            sRunName    = '%s_%s'           % (sRun, sError)
            sOutputDir  = '%s/output/%s_%s' % (sBASE_DIR, sAnalysis, sRun)
            os.makedirs(sOutputDir, exist_ok=True)
            #basic_stat_ABSOLUTE_data (sInputDir)

            ## Load Barcode Data ##
            sBarcodeFile    = '%s/BarcodeTargets/%s_Barcode_Targets_%s.txt' % (sInputDir, sAnalysis, sRun)

            #Top Priority Barcodes ##
            sTopBarcodeFile = '%s/BarcodeTargets/%s_JustBarcode_Top.txt' % (sInputDir, sAnalysis)

            ## Run Analysis ##
            list_sSamples = list(dict_sFiles.keys())
            for sSample in list_sSamples:
                print('Processing %s %s - %s' % (sAnalysis, sRunName, sSample))

                #if sSample != 'PE_off_target_293T_1': continue

                ## HKK Modified By_barcode 200609 ###
                mod_sort_by_barcode   (sDataDir, sSample, sOutputDir, sBarcodeFile)

                sInDir          = '%s/%s/flash/%s'   % (sInputDir, sAnalysis, sSample)
                sFastqTag       = '%s.extendedFrags' % sSample

                #split_fq_file(sInDir, sFastqTag, bTestRun)
                #gzip_fastq_list (sInDir, sFastqTag, bTestRun)
                nLineCnt        = get_line_cnt (sInDir, sFastqTag)
                if sRun.startswith('Endo'):
                    list_sSplitFile = ['%s/%s/%s.extendedFrags.fastq' % (sInDir, sSample, sSample)]
                else:
                    list_sSplitFile = get_split_list (sInDir, sFastqTag)

                mp_sort_by_barcode (nCores, sRun, sSample, sInDir, sOutputDir, sBarcodeFile, list_sSplitFile, sRE, sError, sTopBarcodeFile, nBins)

                #pbs_sort_by_barcode_v2 (sAnalysis, sBarcodeFile, sFastqFile, sOutDir,  sQueue, bTestRun, sTimeLimit)
                #mp_sort_by_barcode_vJustNGS (nCores, sRun, sSample, sInDir, sOutputDir, sBarcodeFile, list_sSplitFile, sRE, sError)
                #sys.exit()

                nSplitNo = 4
                #combine_output_pickle   (sSample, sOutputDir, sRun, sError, sBarcodeFile, list_sSplitFile, nSplitNo)
                #combine_output_freq     (sSample, sOutputDir, sRun, sError, sBarcodeFile, list_sSplitFile, nSplitNo)
                #combine_output_reads    (sSample, sOutputDir, sError, sBarcodeFile, list_sSplitFile, nSplitNo)

                # Combined FASTQ for CRISPResso
                #gzip_fastq(sInDir, sFastqTag, bTestRun)
                #extract_reads_by_readID_for_CRISPResso (nCores, sSample, sInDir, sOutputDir, sFastqTag, sBarcodeFile, sTopBarcodeFile, list_sSplitFile, sError)
                #extract_reads_by_readID_for_CRISPResso_v2 (nCores, sSample, sInDir, sOutputDir, sFastqTag, sBarcodeFile, sTopBarcodeFile, list_sSplitFile, sError, nBins)
                extract_reads_by_readID_for_CRISPResso_v3 (nCores, sSample, sInDir, sOutputDir, sFastqTag, sBarcodeFile, sTopBarcodeFile, list_sSplitFile, sError, nBins)
                mp_run_crispresso (nCores, sAnalysis, sSample, sOutputDir, sBarcodeFile, list_sSplitFile)
                #mp_combined_output_fastq_v3 (nCores, sSample, sOutputDir, sBarcodeFile, list_sSplitFile, sError)
                #combined_output_fastq_vCheckLoss (nCores, sSample, sOutputDir, sBarcodeFile, list_sSplitFile, sError)

                ## Examining Top "Other" Reads ##
                nTop = 50
                #gzip_fastq (sInDir, sFastqTag, bTestRun)
                #extract_reads_by_readID (sSample, sInDir, sFastqTag, sOutputDir, nCores, nTop)
                #extract_reads_by_readID_for_indelsearcher (sSample, sInDir, sFastqTag, sOutputDir, nCores, nTop)
                #combined_output_fastq                     (sSample, sInDir, sFastqTag, sOutputDir, nCores, nTop)
                #analyze_top_barcodes (sSample, sBarcodeFile, sOutputDir, nTop, nLineCnt)
            #loop END: sSample

            ## Combine Analysis Results##
            list_sAnalysis = ['HKK_210906', 'GSY_211008']
            #combined_analysis_output_freq (list_sAnalysis, list_sSamples, sBarcodeFile, sRun, sError)
        #loop END: sError

        #loop END: sRun

    ## Clinvar Data -- Analyzed using local script ##






    '''
    ## PBS Run Analysis ##
    list_sSamples = list(dict_sFiles.keys())
    for sSample in list_sSamples:

        sJobName    = 'PJM.PE.%s' % sSample
        sLogDir     = make_log_dir (sJobName)
        sTmpDir     = make_tmp_dir (sJobName)

        sInDir      = '%s/01_raw_WGSdata/%s' % (sBASE_DIR, sAnalysis)
        sOutDir     = '%s/02_bwa_aln/%s'     % (sBASE_DIR, sAnalysis)
        os.makedirs(sOutDir, exist_ok=True)


        sInDir    = '%s/%s/flash/%s'   % (sInputDir, sAnalysis, sSample)
        sFastqTag = '%s.extendedFrags' % sSample
        #split_fq_file(sInDir, sFastqTag, bTestRun)
        list_sSplitFile = get_split_list (sInDir, sFastqTag)

        #mp_sort_by_barcode (nCores, sSample, sInDir, sOutputDir, sBarcodeFile, list_sSplitFile)
        combine_output     (sSample, sOutputDir, sBarcodeFile, list_sSplitFile)
    #loop END: sSample
    '''

    '''
    ## HGNC Database ##
    sHGNC_File        = '%s/HGNC_genelist.txt' % sREF_DIR
    #dict_sHGNC        = load_HGNC_reference (sHGNC_File)

    ## RefSeq Database ##
    sRefFlat          = '%s/%s_refFlat_step4.txt' % (sGENOME_DIR, sGENOME)
    #dict_cRefGene     = parse_refflat_line(sRefFlat, 'dict', 'Gene')

    ## Genome FASTA ##
    sGenomeFile  = '%s/%s/%s.fa' % (sREF_DIR, sGENOME, sGENOME)
    #cGenome      = cFasta(sGenomeFile)

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

