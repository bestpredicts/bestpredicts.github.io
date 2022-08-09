#!/bin/bash

git init
git remote add origin https://github.com/bestpredicts/bestpredicts.github.io
git checkout -b hexo_bkp
git add .
git commit -m "Initial backup"
git push -f origin hexo_bkp:hexo_bkp
