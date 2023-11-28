from roboflow import Roboflow

rf = Roboflow(api_key="zp3Vyuiz2hhRhgQtz2pF")
project = rf.workspace("mike-caulfild").project("chtozalevetottigr")
dataset = project.version(4).download("folder")