from roboflow import Roboflow
rf = Roboflow(api_key="TR2n5JTcwaRKLAU3sNDd")
project = rf.workspace("mike-caulfild").project("chtozalevetottigr")
dataset = project.version(4).download("folder")

''' при указании project.version(4), скачивается версия датасета,
    разделенная на папки train/val, а при project.version(10) train/val/test '''
