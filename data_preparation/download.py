import SoccerNet
from SoccerNet.Downloader import SoccerNetDownloader

mySoccerNetDownloader = SoccerNetDownloader(LocalDirectory=r"C:\Users\ksost\soccer_env\base_data")

mySoccerNetDownloader.password = "s0cc3rn3t"
mySoccerNetDownloader.downloadGames(files=["Labels-v2.json"], split=["train", "valid", "test"])
# mySoccerNetDownloader.downloadGames(files=["Labels-cameras.json"], split=["train", "valid", "test"])
mySoccerNetDownloader.downloadGames(files=["1_224p.mkv", "2_224p.mkv"], split=["train", "valid", "test"])
# mySoccerNetDownloader.downloadRAWVideo(dataset="SoccerNet-Tracking")