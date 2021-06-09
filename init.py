import os

if not os.path.exists('dataset/'):
    os.mkdir('dataset/')
    os.system('wget http://lfs.aminer.cn/misc/moocdata/data/type_net.zip')
    os.system('wget http://lfs.aminer.cn/misc/moocdata/data/med_mentions.zip')
    os.system('wget http://lfs.aminer.cn/misc/moocdata/data/flowers.zip')
    os.system('wget http://lfs.aminer.cn/misc/moocdata/data/birds.zip')
    os.system('unzip type_net.zip -d dataset/')
    os.system('unzip med_mentions.zip -d dataset/')
    os.system('unzip flowers.zip -d dataset/')
    os.system('unzip birds.zip -d dataset/')