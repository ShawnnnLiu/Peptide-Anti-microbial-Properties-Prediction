import pandas as pd
import re

def clean_sequence(seq):
    """Clean sequence by removing non-standard amino acid codes."""
    if pd.isna(seq):
        return None
    
    # Map circled letters to standard AAs
    circled_map = {
        'Ⓚ': 'K', 'Ⓐ': 'A', 'Ⓡ': 'R', 'Ⓝ': 'N', 'Ⓓ': 'D',
        'Ⓒ': 'C', 'Ⓔ': 'E', 'Ⓠ': 'Q', 'Ⓖ': 'G', 'Ⓗ': 'H',
        'Ⓘ': 'I', 'Ⓛ': 'L', 'Ⓜ': 'M', 'Ⓕ': 'F', 'Ⓟ': 'P',
        'Ⓢ': 'S', 'Ⓣ': 'T', 'Ⓦ': 'W', 'Ⓨ': 'Y', 'Ⓥ': 'V',
        'Ⓧ': 'A',
    }
    
    for circled, standard in circled_map.items():
        seq = seq.replace(circled, standard)
    
    # Remove stapling codes
    seq = re.sub(r'[SR][0-9]+', '', seq)
    seq = re.sub(r'\*', '', seq)
    seq = re.sub(r'\?', '', seq)
    seq = re.sub(r'[0-9]+', '', seq)
    seq = re.sub(r'-', '', seq)
    
    # Keep only standard amino acids
    standard_aa = set('ACDEFGHIKLMNPQRSTVWY')
    cleaned = ''.join([aa for aa in seq.upper() if aa in standard_aa])
    
    return cleaned if len(cleaned) >= 5 else None


# === Convert AMPs ===
print("Converting AMPs...")
amps = pd.read_csv('stapled_amps.csv')

with open('seqs_AMP_stapep.txt', 'w') as f:
    count = 0
    for i, row in amps.iterrows():
        # Use Hiden_Sequence if available (it's already clean)
        if 'Hiden_Sequence' in amps.columns and pd.notna(row['Hiden_Sequence']):
            seq = row['Hiden_Sequence']
        else:
            seq = clean_sequence(row['Sequence'])
        
        if seq and len(seq) >= 5:
            count += 1
            f.write(f'{count:6d} {seq}\n')

print(f'Created: seqs_AMP_stapep.txt ({count} sequences)')


# === Convert Decoys ===
print("\nConverting Decoys...")
decoys = pd.read_csv('stapled_decoys.csv')

with open('seqs_DECOY_stapep.txt', 'w') as f:
    count = 0
    for i, row in decoys.iterrows():
        seq = clean_sequence(row['SEQUENCE'])
        
        if seq and len(seq) >= 5:
            count += 1
            f.write(f'{count:6d} {seq}\n')

print(f'Created: seqs_DECOY_stapep.txt ({count} sequences)')


# === Verify ===
print("\n=== Verification ===")
for fname in ['seqs_AMP_stapep.txt', 'seqs_DECOY_stapep.txt']:
    with open(fname, 'r') as f:
        lines = f.readlines()
    print(f'\n{fname}: {len(lines)} sequences')
    print('First 3:')
    for line in lines[:3]:
        print(f'  {line.rstrip()}')
