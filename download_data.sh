BASEDIR=$(dirname $0)

wget -O PTDB.zip https://www2.spsc.tugraz.at/databases/PTDB-TUG/SPEECH_DATA_ZIPPED.zip
unzip PTDB.zip
mv 'SPEECH DATA' ${BASEDIR}/data/PTDB
rm PTDB.zip

wget -O MDB.tar.gz https://zenodo.org/record/1481172/files/MDB-stem-synth.tar.gz
tar -xvzf MDB.tar.gz
mv MDB-stem-synth ${BASEDIR}/data/MDB
rm MDB.tar.gz