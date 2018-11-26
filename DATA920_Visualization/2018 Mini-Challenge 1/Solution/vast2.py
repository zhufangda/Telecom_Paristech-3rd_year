from skimage import color
from skimage import measure
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re
import datetime as dt
import os
import re
import imageio
from dateutil import parser
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.fftpack import fft
from scipy import signal
from scipy.io import wavfile
from pydub import AudioSegment

def load_data(path, birds_data, kasios_data):
    """
    This function reuse some lines of command of the first assignment to 
    """
    # import the csv file 
    data = pd.read_csv(path + birds_data, parse_dates=[5], dayfirst=True)
    cat_bird = pd.Categorical(data['English_name'],ordered=False)
    i_bp = cat_bird.categories.tolist().index('Rose-crested Blue Pipit')
    nb_categories = len(cat_bird.categories.unique())
    # Add a column with 1 for the Blue pipits else 0
    data['IsRCBP'] = data['English_name'].apply(lambda x: 1 \
                                                if x=="Rose-crested Blue Pipit"\
                                                else 0)
    # clean vocalization
    # all in lower string
    data['Vocalization_type'] = data['Vocalization_type'].apply(str.lower)
    # delete spaces
    data['Vocalization_type'] = data['Vocalization_type'].apply(str.strip)
    # clean the grids 
    # returns -1 if not possible to select a number beetween 0 and 200
    data['X'] = data['X'].apply(clean_grid)
    data['Y'] = data['Y'].apply(clean_grid)
    kasios_records = pd.read_csv(path + kasios_data)
    kasios_records = kasios_records.rename(index=str, columns={" X": "X", " Y":"Y"})
    return data, kasios_records, nb_categories, cat_bird, i_bp

def vectorize(path):
    '''
    This function import the image from the path and vectorize it
    
    parameters:
    --------------------------------------------------------
    path :               str, path of the image file
    --------------------------------------------------------
    '''
    map_1 = imageio.imread(path)
    map_2 = color.colorconv.rgb2grey(map_1)
    map_3 = np.flipud(map_2)
    map_contours = measure.find_contours(map_3, 0.8, fully_connected='high')
    return map_contours

            
def print_map(a, map_contours, title):
    '''
    This function print the map in backgroung
    
    parameters:
    --------------------------------------------------------
    a :               the axe to plot
    map_contours :    background
    title :           str, title of the figure
    --------------------------------------------------------
    '''
    a.set_xlim(0, 200)
    a.set_ylim(0, 200)
    a.grid(color=(0.8, 0.8, 0.8), linestyle='-', zorder=1)
    a.scatter(148, 159, marker="$\u2A3B$", s=1000, color='red', 
           label ="dumping site", zorder=10)
    for contour in map_contours:
        a.plot(contour[:, 1], contour[:, 0], 
               linewidth=1, color='#662506', zorder=2)
    a.set_title(title, fontsize=20)
    a.legend()
    return a

def plotmap(data, kasios_records, list_kasios_bp=[]):
    map_contours = vectorize("LekagulRoadways2018.png")
    fig, ax = plt.subplots(figsize=(10,10))
    # print the map
    print_map(ax, map_contours, "All record locations")
    # plot Blue pipits
    ax.scatter(data.loc[data['IsRCBP']== 1]['X'], 
               data.loc[data['IsRCBP']== 1]['Y'],
               color='blue', alpha=0.7, label="Blue Pipit location")
    # plot others
    ax.scatter(data.loc[data['IsRCBP']== 0]['X'], 
               data.loc[data['IsRCBP']== 0]['Y'],
               color='green', alpha=0.2, label="Other birds location")

    # Add Kasios records with ID
    for i, txt in enumerate(kasios_records['ID'].values):
        if i in list_kasios_bp:
            ax.text(kasios_records['X'].values[i], kasios_records['Y'].values[i], 
                    txt, color='white', ha="center", va="center", 
                    bbox={'pad':0.4, 'boxstyle':'circle', 
                          'edgecolor':'none', 'facecolor':'blue'})
        else:
            ax.text(kasios_records['X'].values[i], kasios_records['Y'].values[i], 
                    txt, color='black', ha="center", va="center", 
                    bbox={'pad':0.4, 'boxstyle':'circle', 
                          'edgecolor':'none', 'facecolor':'orange'})            
    ax.scatter([], [], color='orange', marker='o',s=100, label='Kasios records')
    ax.legend(bbox_to_anchor=(1, 1), labelspacing=1)
    plt.show()



def clean_grid(X, drop_bad_value=False):
    '''
    This function clean the grids
    
    parameters:
    --------------------------------------------------------
    X              :   str, value to check
    drop_bad_value :   booleen (False by default)
                       if True, replace the bad_values by -1
                       if False, try to extract number
    --------------------------------------------------------
    '''
    if type(X) is not int:
        a = int(max(re.findall('\d+', X), key=len))
        if not a:
             return -1
        else:
            if drop_bad_value:
                return -1
            else:
                return a
    else:
        if (0 <= X <= 200):
            return X
        else:
             return -1


def clean_time(time) :
    try: 
        parsed_time = parser.parse(time).hour
    except ValueError :
        parsed_time = 0
    return parsed_time
            
def pause(serie_date, nb):
    for i in range(nb):
        serie_date.append(serie_date[len(serie_date)-1])
    return serie_date


def plot_new_distribub(data):
    fig = plt.subplots(figsize=(12,5))
    ax = data.value_counts().plot(kind='bar', color='#009432', 
                                  label='Other birds', zorder = 2)
    ax.get_children()[16].set_color('#0652DD')
    ax.get_children()[16].set_label('Blue Pipits')
    plt.xticks(rotation=30, ha='right')
    plt.ylabel("# of records", fontsize=12)
    ax.legend()
    ax.grid(which='major', axis='y', linestyle='--', zorder=1)
    plt.title("Distribution of the records by bird category while selecting call, songs and ABC quality", fontsize=16)
    plt.show()


def log_specgram(audio, sample_rate, window_size=20, step_size=10, eps=1e-10):
    """
    This function returns frequencies, times and the log-spectrogramm of the audio file.
    """
    nperseg = int(round(window_size * sample_rate / 1e3))
    noverlap = int(round(step_size * sample_rate / 1e3))
    freqs, times, spec = signal.spectrogram(audio,
                                    fs=sample_rate,
                                    window='hann',
                                    nperseg=nperseg,
                                    noverlap=noverlap,
                                    detrend=False)
    return freqs, times, 10 * np.log10(spec.T.astype(np.float32) + eps)


def plot_magnitude_spectrogram(samples, rate, freqs, times, spectrogram):
    n_sample = len(samples)
    fig = plt.figure(figsize=(14, 8))
    ax1 = fig.add_subplot(211)
    ax1.set_title('Raw wave')
    ax1.set_ylabel('Magnitude')
    ax1.set_xlim(times.min(), times.max())
    ax1.plot(np.linspace(0, n_sample / rate, n_sample), samples, color='#ff7f00', linewidth=0.05)

    ax2 = fig.add_subplot(212)
    ax2.imshow(spectrogram.T, aspect='auto', origin='lower', 
               extent=[times.min(), times.max(), freqs.min(), freqs.max()], cmap='autumn_r')
    ax2.set_title('Spectrogram ')
    ax2.set_ylabel('Freqs in Hz')
    ax2.set_xlabel('Seconds')
    plt.show()

def custom_fft(y, fs):
    """
    Calculate the FFT of the samples y with a rate of fs
    """
    T = 1.0 / fs
    N = y.shape[0]
    yf = fft(y)
    xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
    power = 10 * np.log10(2.0/N * np.abs(yf[0:N//2]))  # FFT is simmetrical, so we take just the first half
    # FFT is also complex, to we take just the real part (abs)
    return xf, power

def plot_fft(fourier, power):
    plt.figure(figsize=(14,6))
    plt.plot(fourier/1000, power, color='#ff7f00', linewidth=0.05)
    plt.xlabel('Frequency (kHz)', fontsize = 16)
    plt.ylabel('Power (dB)', fontsize = 16)
    plt.show()

    
def get_sequences(spectrogram, l_sequence=100, tolerance=0.3):
    """
    split a spectrogram into sequences of lenght l_sequnce, 
            and overlapp the last one if there is 
            enough information (len(last) > tolerance * l_sequence)
    """
    nb_frames = len(spectrogram)
    list_sequences = []
    nb_sequences = nb_frames // l_sequence

    if nb_sequences != 0:
        for j in range(nb_sequences - 1):
            list_sequences.append(spectrogram[j * l_sequence: (j+1) * l_sequence])
    else:
        # we have less than n_frames_per_sequence in the frame...
        # overlap the last sequence if there is sufficient information
        if (nb_frames % l_sequence > l_sequence * tolerance):
            list_sequences.append(spectrogram[-l_sequence:])
    return list_sequences

def plot_spectrogram(freqs, times, spectrogram):
    fig = plt.figure(figsize=(14, 4))
    plt.imshow(spectrogram.T, extent=[0, 100, freqs.min(), freqs.max()], aspect='auto', origin='lower', cmap='autumn_r')
    plt.title('Spectrogram ')
    plt.ylabel('Freqs in Hz')
    plt.xlabel('Nb frames')
    plt.show()

    
def edges(e_min, e_max, nb_bins):
    return np.arange(e_min, e_max + (e_max-e_min) / nb_bins, (e_max-e_min) / nb_bins)

def get_features(category, sequence, freq_red,
                 f_mean_edges = edges(1000, 10000, 100),
                 f_std_edges = edges(1000, 4000, 50),
                 f_mode_edges = edges(1000, 10000, 100),
                 f_delta_mode_edges = edges(-2000, 2000, 50),
                 n_frames_per_sequence=100):
    """
    """
    l_sequence = len(sequence)
    if l_sequence < 100:
        for i in range(l_sequence, 100):
            sequence.append(sequence[l_sequence-1])
            
    f_mean = [np.sum(freqs_red * sequence[i]) for i in range(n_frames_per_sequence)]
    f_std = [(np.abs(np.sum(sequence[i] * (freqs_red -f_mean[i]) **2 ))) ** (0.5) for i in range(n_frames_per_sequence)]
    f_mode = [freqs_red[np.argmax(sequence[i])] for i in range(n_frames_sequence)]
    f_delta_mode = np.roll(f_mode, -1) - f_mode

    sequence_histogram1, _, _ = np.histogram2d(f_mean, f_std, bins=(f_mean_edges, f_std_edges))
    features1 = sequence_histogram1.T.flatten()

    sequence_histogram2, _ = np.histogram(f_mode, bins=f_mode_edges)
    features2 = sequence_histogram2

    sequence_histogram3, _, _ = np.histogram2d(f_mode, f_delta_mode, bins=(f_mode_edges, f_delta_mode_edges))
    features3 = sequence_histogram3.T.flatten()
    
    return np.concatenate([[category], features1, features2, features3])


def plot2_features(title, H1, H2, H3, freq_red,
                   f_mean_edges = edges(1000, 10000, 100),
                   f_std_edges = edges(1000, 4000, 50),
                   f_mode_edges = edges(1000, 10000, 100),
                   f_delta_mode_edges = edges(-2000, 2000, 50),
                   n_frames_per_sequence=100):
    """
    """
    fig, axs = plt.subplots(1, 3, figsize=(14, 5))
    axs[0].imshow(H1.T,interpolation='nearest', origin='low', extent=[f_mean_edges[0], f_mean_edges[-1], f_std_edges[0], f_std_edges[-1]])
    axs[0].set_xlabel("$f_{mean}$", fontsize=16)
    axs[0].set_ylabel("$f_{std}$", fontsize=16)
    axs[0].set_xlim(f_mean_edges[0], f_mean_edges[-1])
    axs[0].set_ylim(f_std_edges[0], f_std_edges[-1])
    axs[0].set_aspect(3)

    axs[1].stem(f_mode_edges[:-1], H2)
    axs[1].set_xlabel("$f_{mode}$", fontsize=16)
    axs[1].set_ylabel("nb of values", fontsize=12)
    axs[1].set_xlim(f_mode_edges[0], f_mode_edges[-1])

    axs[2].imshow(H3.T, interpolation='nearest', origin='low', extent=[f_mode_edges[0], f_mode_edges[-1], f_delta_mode_edges[0], f_delta_mode_edges[-1]])
    axs[2].set_xlabel("$f_{mode}$", fontsize=16)
    axs[2].set_ylabel("$f_{\Delta mode}$", fontsize=16)
    axs[2].set_xlim(f_mode_edges[0], f_mode_edges[-1])
    axs[2].set_ylim(f_delta_mode_edges[0], f_delta_mode_edges[-1])
    axs[2].yaxis.set_label_position("right")
    axs[2].set_aspect(2)
    plt.suptitle(title, fontsize=24)
    plt.show()
    
    
def get_spectrogram(file, data_sounds, cat_bird, window_size=20, step_size=10):
    """
    This function open a wav file and return the desired normalized spectrogram
    inputs : 
        str file : name of the sound file (wav)
        int window_size : lenght of a frame in ms
        int step_size : for the overlapping
    output : 
        int category : category of the bird according to the name of the file
        array spectogram_normalized : the normalized spectrogram
        list freqs_red : list of frequences
    """
    
    record_id, _ = re.split('.wav',file)
    bird_category = data_sounds.loc[data_sounds['File ID']==int(record_id)]['English_name'].values[0]
    category = cat_bird.categories.tolist().index(bird_category)
    
    rate, frames = wavfile.read("Sounds/Out/" + file)
    n_frames = len(frames)
    
    # if stereo, keep only one channel
    if frames.ndim == 2:
        frames = frames[:,0]
    
    # Step 1 : select the most energetic frames
    # mean of the 1% most energetic frames 
    mean_high_NRJ_frames = np.mean(heapq.nlargest(n_frames//100, np.abs(frames)))
    # select only frames of that recording that have power of at least 0.25 of the estimated highest level
    frames = np.array([frame for i_frame, frame in enumerate(frames) if np.abs(frame) > 0.25 * mean_high_NRJ_frames])
    # update nb frames
    n_frames = len(frames)
    
    # Step 2 : split into 20ms chunks and get spectrogram
    window_size=20
    step_size=10
    freqs, times, spectrogram = log_specgram(frames, rate, window_size=window_size, step_size=step_size)
    
    # Step 3 : keep only 1-10KHz
    # keep only fq between 1kHz and 10 kHz
    indices = [i for i, fq in enumerate(freqs) if fq>1000 and fq<10000]   
    spectrogram_red = spectrogram[:,indices]
    freqs_red = freqs[indices]
    # Step 4: normalize
    spectogram_normalized = normalize(spectrogram_red, norm="l1", axis=1)
    
    return category, spectogram_normalized, freqs_red    
    
def get_spectrogram_kasios(krecord_id, window_size=20, step_size=10):
    """
    This function open a wav file and return the desired normalized spectrogram
    inputs : 
        str file : name of the sound file (wav)
        int window_size : lenght of a frame in ms
        int step_size : for the overlapping
    output : 
        int category : category of the bird according to the name of the file
        array spectogram_normalized : the normalized spectrogram
        list freqs_red : list of frequences
    """

    rate, frames = wavfile.read("Sounds_Kasios/Out/" + str(krecord_id) + ".wav")
    n_frames = len(frames)
    
    # if stereo, keep only one channel
    if frames.ndim == 2:
        frames = frames[:,0]
    
    # Step 1 : select the most energetic frames
    # mean of the 1% most energetic frames 
    mean_high_NRJ_frames = np.mean(heapq.nlargest(n_frames//100, np.abs(frames)))
    # select only frames of that recording that have power of at least 0.25 of the estimated highest level
    frames = np.array([frame for i_frame, frame in enumerate(frames) if np.abs(frame) > 0.25 * mean_high_NRJ_frames])
    # update nb frames
    n_frames = len(frames)
    
    # Step 2 : split into 20ms chunks and get spectrogram
    window_size=20
    step_size=10
    freqs, times, spectrogram = log_specgram(frames, rate, window_size=window_size, step_size=step_size)
    
    # Step 3 : keep only 1-10KHz
    # keep only fq between 1kHz and 10 kHz
    indices = [i for i, fq in enumerate(freqs) if fq>1000 and fq<10000]   
    spectrogram_red = spectrogram[:,indices]
    freqs_red = freqs[indices]
    # Step 4: normalize
    spectogram_normalized = normalize(spectrogram_red, norm="l1", axis=1)
    
    return spectogram_normalized, freqs_red    
    
    
def get_sequences(spectrogram, l_sequence=100, tolerance=0.3):
    """
    split the spectrogram in spectrograms of size l_sequence (default 100)
        and keep the last one if its lenght is longer than tolerance * l_sequence
            (overlap with the previous one)
    """
    nb_frames = len(spectrogram)
    list_sequences = []
    nb_sequences = nb_frames // l_sequence

    if nb_sequences != 0:
        for j in range(nb_sequences):
            list_sequences.append(spectrogram[j * l_sequence: (j+1) * l_sequence])
    else:
        # we have less than n_frames_per_sequence in the frame...
        # overlap the last sequence if there is sufficient information
        if (nb_frames % l_sequence > l_sequence * tolerance):
            list_sequences.append(spectrogram[-l_sequence:])

    return list_sequences

def get_sequences_per_category(all_spectrograms, nb_categories=19, lenght_sequence=100, tolerance=0.3):
    """
    This function transform a list of spectrograms
    ---------------------------------------------------------------------------
    inputs:
        all_spectrograms : list of spectrograms listed by category
        lenght_sequence : number of frame per sequence
        tolerance : keep only sequences 
                        with a lenght at least tolerance * lenght_sequence
    ---------------------------------------------------------------------------
    outputs:
        all_sequences : list of sequences listed by category
    """
    
    # concatene all spectrogram per category
    list_unique_spec = [[] for _ in range(nb_categories)]
    for category in range(nb_categories):
        if len(all_spectrograms[category])==0:
            continue
        list_unique_spec[category] = np.vstack(all_spectrograms[category])
    
    all_sequences = [get_sequences(list_unique_spec[category],
                                           l_sequence=lenght_sequence,
                                           tolerance=tolerance) 
                     for category in range(nb_categories)]

    return all_sequences

    
def get_features2(category, sequence, freq_red,
                  n_frames_per_sequence=100):
    """
    """
    l_sequence = len(sequence)
    if l_sequence < 100:
        for i in range(l_sequence, 100):
            sequence.append(sequence[l_sequence-1])
            
    f_mean = [np.sum(freqs_red * sequence[i]) for i in range(n_frames_per_sequence)]
    f_std = [(np.abs(np.sum(sequence[i] * (freqs_red -f_mean[i]) **2 ))) ** (0.5) for i in range(n_frames_per_sequence)]
    f_mode = [freqs_red[np.argmax(sequence[i])] for i in range(n_frames_sequence)]
    f_delta_mode = np.roll(f_mode, -1) - f_mode

    return np.concatenate([[category], f_mean, f_std, f_mode, f_delta_mode])    
    
def plot3_features(title, H1, H2, H3, freq_red, axs,
                   f_mean_edges = edges(1000, 10000, 100),
                   f_std_edges = edges(1000, 4000, 50),
                   f_mode_edges = edges(1000, 10000, 100),
                   f_delta_mode_edges = edges(-2000, 2000, 50),
                   n_frames_per_sequence=100):
    """
    """
    axs[0].imshow(H1.T,interpolation='nearest', origin='low', extent=[f_mean_edges[0], f_mean_edges[-1], f_std_edges[0], f_std_edges[-1]])
    axs[0].set_xlim(f_mean_edges[0], f_mean_edges[-1])
    axs[0].set_ylim(f_std_edges[0], f_std_edges[-1])
    axs[0].set_aspect(3)
    axs[1].stem(f_mode_edges[:-1], H2)
    axs[1].set_xlim(f_mode_edges[0], f_mode_edges[-1])
    axs[1].set_title(title, fontsize=16)
    #axs[1].set_aspect(2)
    axs[2].imshow(H3.T, interpolation='nearest', origin='low', extent=[f_mode_edges[0], f_mode_edges[-1], f_delta_mode_edges[0], f_delta_mode_edges[-1]])
    axs[2].set_xlim(f_mode_edges[0], f_mode_edges[-1])
    axs[2].set_ylim(f_delta_mode_edges[0], f_delta_mode_edges[-1])
    axs[2].yaxis.set_label_position("right")
    axs[2].set_aspect(2)    
    
    
def plot_heatmap(data, row_labels, col_labels, title="",
                    cbar_kw={}, cbarlabel="", txt=False, **kwargs):
    """
    Plot a heatmap from a numpy array and two lists of labels.

    Arguments:
        data       : A 2D numpy array of shape (N,M)
        row_labels : A list or array of length N with the labels
                     for the rows
        col_labels : A list or array of length M with the labels
                     for the columns
    Optional arguments:
        ax         : A matplotlib.axes.Axes instance to which the heatmap
                     is plotted. If not provided, use current axes or
                     create a new one.
        cbar_kw    : A dictionary with arguments to
                     :meth:`matplotlib.Figure.colorbar`.
        cbarlabel  : The label for the colorbar
    All other arguments are directly passed on to the imshow call.
    """
    fig, ax = plt.subplots(figsize=(10,10))
    
    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    
    cbar = ax.figure.colorbar(im, cax=cax, **cbar_kw)
    
    
    
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    #ax.tick_params(top=True, bottom=False,
    #               labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    
    # Loop over data dimensions and create text annotations.
    if txt==True:
        for i in range(len(col_labels)):
            for j in range(len(row_labels)):
                text = ax.text(j, i, np.round(data[i, j],2),
                               ha="center", va="center", color="black")

    
    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_title(title, fontsize=30)
    fig.tight_layout()
    plt.show()    
    
    
js_getResults = """<div id="d3-container"></div>

<style>

.node {stroke: #fff; stroke-width: 1.5px;}
.link {stroke: #999; stroke-opacity: .6;}

text {
  font: 14px sans-serif;
  pointer-events: none;
  text-shadow: 0 1px 0 #fff, 1px 0 0 #fff, 0 -1px 0 #fff, -1px 0 0 #fff;
}

</style>


<script>

// We load the latest version of d3.js from the Web.
require.config({paths: {d3: "https://d3js.org/d3.v3.min"}});
require(["d3"], function(d3) {
    
    // Parameter declaration, the height and width of our viz.
    var width = 800,
        height = 800;

    // Colour scale for node colours.
    var color = d3.scale.category10();

    // We create a force-directed dynamic graph layout.
    // D3 has number of layouts - refer to the documentation.
    var force = d3.layout.force()
        .charge(-1000)
        .linkDistance(150)
        .size([width, height]);

    // We select the < div> we created earlier and add an <svg> container.
    // SVG = Scalable Vector Graphics
    var svg = d3.select("#d3-container").select("svg")
    if (svg.empty()) {
        svg = d3.select("#d3-container").append("svg")
                    .attr("width", width)
                    .attr("height", height);
    }
        
    // We load the JSON network file.
    d3.json("graphsim.json", function(error, graph) {
        // Within this block, the network has been loaded
        // and stored in the 'graph' object.
        
        force.nodes(graph.nodes)
            .links(graph.links)
            .start();

        var link = svg.selectAll(".link")
            .data(graph.links)
            .enter().append("line")
            .attr('stroke-width', function(d) { return d.weight; })
            .attr("class", "link");

        var node = svg.selectAll(".node")
            .data(graph.nodes)
            .enter().append("g")
            .attr("class", "node")
            .call(force.drag)
            .on('click', connectedNodes);
        node.append("circle")
            .attr("r", 15)  // radius
            .style("fill", function(d) {return  d.id==16 ?'#0652DD':'#009432';})
            
       
        node.append("text")
            .attr("dx", function(d){return d.isKasios==0 ? 10 : -5;})
            .attr("dy", ".15em")
            .text(function(d) { return d.category ;})
            .attr("stroke", "black");
            
        node.append("title")
            .text(function(d) { return d.category ;});
       
        force.on("tick", function() {
            link.attr("x1", function(d) { return d.source.x; })
                .attr("y1", function(d) { return d.source.y; })
                .attr("x2", function(d) { return d.target.x; })
                .attr("y2", function(d) { return d.target.y; });
            node.attr("cx", function(d) { return d.x; })
                .attr("cy", function(d) { return d.y; });
                d3.selectAll("circle").attr("cx", function (d) {
                    return d.x;
                })
                    .attr("cy", function (d) {
                    return d.y;
                });
                d3.selectAll("text").attr("x", function (d) {
                    return d.x;
                })
                    .attr("y", function (d) {
                    return d.y;
                });
            });
        
        var toggle = 0;


        var linkedByIndex = {};
        for (var i = 0; i < graph.nodes.length; i++) {
            linkedByIndex[i + "," + i] = 1;
        };
        graph.links.forEach(function (d) {
            linkedByIndex[d.source.index + "," + d.target.index] = 1;
        });
        function neighboring(a, b) {
            return linkedByIndex[a.index + "," + b.index];
        }
     
        function connectedNodes() {
            if (toggle == 0) {
                var d = d3.select(this).node().__data__;
                
                link.style("opacity", function (o) {
                    return d.id==o.source.index | d.index==o.target.index ? 1 : 0.8;
                });
                link.style("stroke-width", function (o) {
                    return d.index==o.source.index | d.index==o.target.index ?  o.weight : 0.8;
                });
                node.style("opacity", function (o) {
                    return neighboring(d, o) | neighboring(o, d) ? 1 : 0.3;
                });
                //Reset the toggle.
                toggle = 1;
            } else {
                //Restore everything back to normal
                node.style("opacity", 1);
                link.style("opacity", 1);
                link.style("stroke-width",function(d) { return  d.weight; });
                toggle = 0;
            }
        } 

        
    });
});
</script>
"""
    
    
js_getResults2 = """<div id="d3-container2"></div>

<style>

.node {stroke: #fff; stroke-width: 1.5px;}

text {
  font: 12px sans-serif;
  pointer-events: none;
  
}

</style>


<script>

// We load the latest version of d3.js from the Web.
require.config({paths: {d3: "https://d3js.org/d3.v3.min"}});
require(["d3"], function(d3) {
    
    // Parameter declaration, the height and width of our viz.
    var width = 800,
        height = 800;


    // We create a force-directed dynamic graph layout.
    // D3 has number of layouts - refer to the documentation.
    var force = d3.layout.force()
        .charge(-1000)
        .linkDistance(125)
        .size([width, height]);
    
    
    // We select the < div> we created earlier and add an <svg> container.
    // SVG = Scalable Vector Graphics
    var svg = d3.select("#d3-container2").select("svg")
    if (svg.empty()) {
        svg = d3.select("#d3-container2").append("svg")
                    .attr("width", width)
                    .attr("height", height)
    }
        
    // We load the JSON network file.
    d3.json("graphsim2.json", function(error, graph) {
        // Within this block, the network has been loaded
        // and stored in the 'graph' object.

        
        force.nodes(graph.nodes)
            .links(graph.links)
            .start();
        
        var link = svg.selectAll(".link")
            .data(graph.links)
            .enter().append("line")
            .attr('stroke-width', function(d) { return d.weight})
            .attr("class", "link")
            .style("stroke", function(d) { return d.stype==0 ? '#3e2723': '#212121'});

        var node = svg.selectAll(".node")
            .data(graph.nodes)
            .enter().append("g")
            .attr("class", "node")
            .call(force.drag)
            .on('click', connectedNodes);
        node.append("circle")
            .attr("r", 15)  // radius
            .style("fill", function(d) {return  colornode(d.id)})
       
        node.append("text")
            .attr("dx", function(d){return d.isKasios==0 ? 10 : -7;})
            .attr("dy", ".15em")
            .text(function(d) { return d.category ;})
            .attr("stroke", "black");
            
        node.append("title")
            .text(function(d) { return d.tip ;});

        force.on("tick", function() {
            link.attr("x1", function(d) { return d.source.x; })
                .attr("y1", function(d) { return d.source.y; })
                .attr("x2", function(d) { return d.target.x; })
                .attr("y2", function(d) { return d.target.y; });
            node.attr("cx", function(d) { return d.x; })
                .attr("cy", function(d) { return d.y; });
                d3.selectAll("circle").attr("cx", function (d) {
                    return d.x;
                })
                    .attr("cy", function (d) {
                    return d.y;
                });
                d3.selectAll("text").attr("x", function (d) {
                    return d.x;
                })
                    .attr("y", function (d) {
                    return d.y;
                });
            });
        
        
        function colornode(a) {
            if (a == 16)
                {return '#0652DD';}
            if (a > 18)
                {return '#FFC107';}
            if (a == 19)
                {return '#FF9800';}
            if (a == 24)
                {return '#FF9800';}
            if (a == 29)
                {return '#FF9800';}
            if (a == 33)
                {return '#FF9800';}
            else
                {return '#009432';}
        }
        
        var toggle = 0;

        var linkedByIndex = {};
        for (var i = 0; i < graph.nodes.length; i++) {
            linkedByIndex[i + "," + i] = 1;
        };
        graph.links.forEach(function (d) {
            linkedByIndex[d.source.index + "," + d.target.index] = 1;
        });
        function neighboring(a, b) {
            return linkedByIndex[a.index + "," + b.index];
        }
        
        function connectedNodes() {
            if (toggle == 0) {
                var d = d3.select(this).node().__data__;
                
                link.style("opacity", function (o) {
                    return d.id==o.source.index | d.index==o.target.index ? 1 : 0.8;
                });
                link.style("stroke-width", function (o) {
                    return d.index==o.source.index | d.index==o.target.index ? o.weight : 0.8;
                });
                node.style("opacity", function (o) {
                    return neighboring(d, o) | neighboring(o, d) ? 1 : 0.3;
                });
                //Reset the toggle.
                toggle = 1;
            } else {
                //Restore everything back to normal
                node.style("opacity", 1);
                link.style("opacity", 1);
                link.style("stroke-width",function(d) { return d.weight; });
                toggle = 0;
            }
        } 

        
    });
});
</script>
"""
    
    