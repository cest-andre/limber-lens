import matplotlib.pyplot as plt
import numpy as np


def plot_scores(scores_path):
    gt_mean = 26.176132202148438
    intact_mean = 21.0178165435791
    shuffled_mean = 12.953747749328613

    labels = ()
    score_means = {'Img Toks': (), 'Txt Toks': (), 'Img Toks+Txt Attn': ()}
    for n in range(28):
        firstn_img = np.mean(np.load(f'{scores_path}/first{n+1}_img_out_ablate_scores.npy'))
        firstn_txt = np.mean(np.load(f'{scores_path}/first{n+1}_txt_out_ablate_scores.npy'))
        firstn_attn = np.mean(np.load(f'{scores_path}/first{n+1}_img_out_img2txt_attn_ablate_scores.npy'))

        labels += (f'{n+1}',)
        score_means['Img Toks'] += ((firstn_img/gt_mean)*100,)
        score_means['Txt Toks'] += ((firstn_txt/gt_mean)*100,)
        score_means['Img Toks+Txt Attn'] += ((firstn_attn/gt_mean)*100,)

    x = np.arange(len(labels))
    width = .25
    mult = 0
    fig, ax = plt.subplots(layout='constrained')
    for attr, mean in score_means.items():
        offset = width * mult
        rects = ax.bar(x+offset, mean, width, label=attr)
        # ax.bar_label(rects, padding=3)
        mult += 1

    ax.set_ylabel('Percent GT CLIPScore (26.18)')
    ax.set_xlabel('Ablate First N Layers')
    ax.set_title('Layer Ablation Effects on Top-16 Logit CLIPScore')
    ax.set_xticks(x+width, labels)
    ax.axhline(y=(intact_mean / gt_mean)*100, color='b', label='Intact')
    ax.axhline(y=(shuffled_mean / gt_mean)*100, color='r', label='Shuffled')
    ax.legend(loc='upper right', ncols=3)
    ax.set_ylim([45, 85])
    plt.show()


def plot_norm_scores():
    gt_scores = np.load(r'E:\Documents\Research\mats\sprint_project\clipscores\gt_scores.npy')
    gt_mean = np.mean(gt_scores)
    intact_mean = np.mean(np.load(r'E:\Documents\Research\mats\sprint_project\clipscores\no_ablate_scores.npy'))

    intact_norm = np.mean(np.load(r'E:\Documents\Research\mats\sprint_project\clipscores\reduced_norm_scores.npy'))
    first7_norm = np.mean(np.load(r'E:\Documents\Research\mats\sprint_project\clipscores\reduced_norm_first7_img_ablate_scores.npy'))
    first14_norm = np.mean(np.load(r'E:\Documents\Research\mats\sprint_project\clipscores\reduced_norm_first14_img_ablate_scores.npy'))
    first21_norm = np.mean(np.load(r'E:\Documents\Research\mats\sprint_project\clipscores\reduced_norm_first21_img_ablate_scores.npy'))
    first28_norm = np.mean(np.load(r'E:\Documents\Research\mats\sprint_project\clipscores\reduced_norm_first28_img_ablate_scores.npy'))

    labels = ('Intact', 'First 7', 'First 14', 'First 21', 'All')
    score_means = {'Img Tokens': ((intact_norm / gt_mean)*100, (first7_norm / gt_mean)*100, (first14_norm / gt_mean)*100, (first21_norm / gt_mean)*100, (first28_norm / gt_mean)*100)}
    x = np.arange(len(labels))
    width = .25
    # mult = 0
    fig, ax = plt.subplots(layout='constrained')
    for attr, mean in score_means.items():
        # offset = width * mult
        rects = ax.bar(x, mean, width)
        ax.bar_label(rects, padding=3)
        # mult += 1

    ax.set_ylabel('Percent of GT CLIPScore (26.18)')
    ax.set_title('Layer Ablation Effects on Reduced-Norm Image Tokens\nTop-16 Logit CLIPScore')
    ax.set_xticks(x, labels)
    ax.axhline(y=(intact_mean / gt_mean)*100, color='r', label='Intact')
    ax.legend(loc='lower right')
    plt.show()


def plot_ioi():
    pass