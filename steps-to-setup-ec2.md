
# Setup Python
```
sudo dnf groupinstall "Development Tools" -y
sudo dnf install -y \
  openssl-devel bzip2-devel libffi-devel zlib-devel \
  readline-devel sqlite-devel xz-devel tk-devel \
  ncurses-devel gdbm-devel

curl https://pyenv.run | bash

cat > ~/.bash_profile << 'EOF'
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init --path)"

if [ -f ~/.bashrc ]; then
    . ~/.bashrc
fi
EOF

cat > ~/.bashrc << 'EOF'
# Load system defaults
if [ -f /etc/bashrc ]; then
    . /etc/bashrc
fi

# Initialize pyenv for interactive shells
if command -v pyenv 1>/dev/null 2>&1; then
    eval "$(pyenv init -)"
fi
EOF

pyenv --version
```

```
pyenv install --list | grep 3.11
```
Choose one from the list, then:
```
pyenv install 3.11.x
```
Then:
```
pyenv global 3.11.x
```
Verify:
```
python --version
which python
```

# Setup env
```
python -m venv .venv
source .venv/bin/activate

pip install "pip<24.1"
pip install "setuptools<70"
pip install wheel setuptools_scm

pip install -r requirements.txt --no-build-isolation
```

# Setup NSD
```
cat nsd_file_list.txt | \
xargs -P 16 -I {} bash -c '
clean="${0#nsd/}";
mkdir -p "$(dirname "$0")";
wget -q -c https://natural-scenes-dataset.s3.amazonaws.com/$clean -O "$0"
' {}
```
# Setup COCO Captions
```
mkdir nsd/nsddata_stimuli/stimuli/nsd/annotations/
cd nsd/nsddata_stimuli/stimuli/nsd/annotations/

wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip

cd ..
mv annotations/annotations/*.json annotations/
rm -r annotations/annotations
```