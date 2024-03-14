from simple_image_download import simple_image_download as sim

my_downloader = sim.Downloader()

my_downloader.directory = 'my_dir/'
# Change File extension type
my_downloader.extensions = '.jpeg'
print(my_downloader.extensions)

my_downloader.download('Auto', limit=100)
my_downloader.download('Bus', limit=100)
