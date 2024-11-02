import os

from pytube import YouTube
import re
import json
from tqdm import tqdm
from moviepy.editor import VideoFileClip
import shutil

from pytube.innertube import _default_clients



# this is a workaround for a current bug in pytube.
# In the future, it may br possible (or necessary) to remove it
_default_clients["ANDROID"]["context"]["client"]["clientVersion"] = "19.08.35"
_default_clients["IOS"]["context"]["client"]["clientVersion"] = "19.08.35"
_default_clients["ANDROID_EMBED"]["context"]["client"]["clientVersion"] = "19.08.35"
_default_clients["IOS_EMBED"]["context"]["client"]["clientVersion"] = "19.08.35"
_default_clients["IOS_MUSIC"]["context"]["client"]["clientVersion"] = "6.41"
_default_clients["ANDROID_MUSIC"] = _default_clients["ANDROID_CREATOR"]

def sanitize_filename(filename):
    '''
    formats video name (or really any string) into a valid filename
    Args:
        filename: original name

    Returns: sanitized filename

    '''
    # Replace spaces with underscores
    filename.strip()
    sanitized = filename.replace(' ', '_')

    # Remove invalid characters using a regular expression
    sanitized = re.sub(r'''[\/:*?"<>|,()'\[\]]''', '', sanitized)
    sanitized=re.sub(r'_{2,}', '_', sanitized)

    return sanitized

def download(link, path):
    '''
    downloads youtube video
    Args:
        link: link to youtube video to download
        path: path to save video to

    Returns:None

    '''
    youtubeObject = YouTube(link)
    title=sanitize_filename(youtubeObject.title)
    youtubeObject= youtubeObject.streams.get_highest_resolution()

    youtubeObject.download(output_path=path, filename=title+'.mp4')

    return os.path.join(path, title+'.mp4')

def make_dataset(path='MAS'):
    '''

    Args:
        path: path to create MAS dataset at

    Returns: None

    '''
    os.makedirs(path+'/temp', exist_ok=True)
    os.makedirs(path + '/clips', exist_ok=True)
    shutil.copy('jsons/annotations.json', os.path.join(path, 'annotations.json'))
    with open('jsons/videos.json') as f:
        videos= json.load(f)

    for video in tqdm(videos):
        try:
            link=videos[video][0]
            clips=videos[video][1]

            video_path=download(link, os.path.join(path, 'temp'))
            video_obj=VideoFileClip(video_path)

            for start, duration, title in clips:
                clip=video_obj.subclip(start, start+duration)
                clip.to_videofile(f'{path}/clips/{title}', verbose=False, logger=None)

            video_obj.close()

            os.remove(video_path)
        except Exception as e:
            print(f'could not retrieve video {video}:')
            print(e)


if __name__=='__main__':
    make_dataset()