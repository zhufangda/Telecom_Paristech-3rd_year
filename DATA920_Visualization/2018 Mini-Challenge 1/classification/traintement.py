from pydub import AudioSegment
import scipy.io.wavfile
import os
import glob
def parser(path):
    '''
    参数化图片

    '''
    rate, data = scipy.io.wavfile.read(path)
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=rate, n_mfcc=40).T)
    name, file_id = re.search('([\w|-]*)-(\d+)\.\w*$', path).groups()
    return [name, file_id, rate, mfcc]

if __name__ == '__main__':
    # cores = multiprocessing.cpu_count()
    # pool = Pool(processes=cores)
    # file_list = list(glob.glob('../ALL_BIRDS_wav/*.wav'))
    # table = []
    # for res in tqdm.tqdm(pool.imap_unordered(parser, file_list), total=len(file_list)):
    #     table.append(res)
    print(list(glob.glob('../ALL_BIRDS_wav/*.wav')))


    print(parser('../ALL_BIRDS_wav\\Orange-Pine-Plover-163250.wav'))

