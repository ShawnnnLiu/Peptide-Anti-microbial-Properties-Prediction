'''
Generating all subsequences of specified length range from each mother sequence held within <motherSeqs> file using sliding windows

IN:
<motherSeqsFile>     = one peptide sequence per line
<minLen>             = minimum window length
<maxLen>             = maximum window length

OUT:
subSeqs_<i>.txt		 = i=1..(# seqs in <motherSeqsFile>), all sliding window subsequences of length [minLeng,maxLen]


'''

## imports
import os, re, sys, time
import random, math

import numpy as np
import numpy.matlib

import scipy.optimize
import scipy.stats


## methods

# usage
def _usage():
	print 'USAGE: %s <motherSeqsFile> <OutFolder> <minLen> <maxLen>' % sys.argv[0]
	print '       <motherSeqsFile>     = one peptide sequence per line'
	print '       <OutFolder>          = output folder'
	print '       <minLen>             = minimum window length'
	print '       <maxLen>             = maximum window length'

def unique(a):
	''' return the list with duplicate elements removed '''
	return list(set(a))

def intersect(a, b):
    ''' return the intersection of two lists '''
    return list(set(a) & set(b))

def union(a, b):
    ''' return the union of two lists '''
    return list(set(a) | set(b))



## main

# reading args and error checking
if len(sys.argv) != 5:
	_usage()
	sys.exit(-1)

motherSeqsFile = str(sys.argv[1])
outputFolder = str(sys.argv[2])
minLen = int(sys.argv[3])
maxLen = int(sys.argv[4])


# loading mother sequences

print('')
print('Loading mother sequences from %s...' % (motherSeqsFile))

motherSeqs=[]
with open(motherSeqsFile,'r') as fin:
	for line in fin:
		motherSeqs.append(line.strip())

print('Mother sequences loaded')
print('')


# for each mother sequence generating all peptide subsequences within range [minLen,maxLen]
for ii in range(0,len(motherSeqs)):

	index = 0

	print "Processing mother sequence %d of %d:\n %s" % (ii+1, len(motherSeqs), motherSeqs[ii])

	with open(outputFolder+'/subSeqs_' + str(ii+1) + '.txt','w') as fout:

		if minLen>len(motherSeqs[ii]):
			minLen = len(motherSeqs[ii])

		for winLen in range(minLen,min(maxLen,len(motherSeqs[ii]))+1):

			for startPos in range(0,len(motherSeqs[ii])-winLen+1):

				index = index + 1
				fout.write(str(index) + '	' + motherSeqs[ii][startPos:startPos+winLen])
				fout.write('\n')

print('')

print('DONE!')
print('')
