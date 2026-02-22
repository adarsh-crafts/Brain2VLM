## Local -> EC2
### setup python
```
sudo dnf groupinstall "Development Tools" -y 
sudo dnf install openssl-devel bzip2-devel libffi-devel zlib-devel readline-devel sqlite-devel xz-devel 
tk-devel -y 
cd /usr/src 
sudo wget https://www.python.org/ftp/python/3.10.13/Python-3.10.13.tgz 
sudo tar -xzf Python-3.10.13.tgz 
sudo ./configure --enable-optimizations 
sudo make altinstall
```

### setup NSD
```
cat nsd_file_list.txt | \
xargs -P 16 -I {} bash -c '
clean="${0#nsd/}";
mkdir -p "$(dirname "$0")";
wget -q -c https://natural-scenes-dataset.s3.amazonaws.com/$clean -O "$0"
' {}
```
```
mkdir nsd/nsddata_stimuli/stimuli/nsd/annotations/
cd nsd/nsddata_stimuli/stimuli/nsd/annotations/

wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip

cd ..
mv annotations/annotations/*.json annotations/
rm -r annotations/annotations
```