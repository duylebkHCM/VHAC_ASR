cat training30h_100h.txt |\
python preprocess.py |\
./kenlm/bin/lmplz -o 3 > bible.arpa
./kenlm/bin/build_binary bible.arpa weight/bible.binary