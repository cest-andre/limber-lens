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


def plot_norm_scores(scores_path):
    gt_mean = 26.176132202148438
    intact_mean = 21.0178165435791
    shuffled_mean = 12.953747749328613
    intact_normed_mean = 13.06905746459961

    labels = ()
    score_means = {'Img Toks': (), 'Img Toks+Txt Attn': ()}
    for n in range(28):
        firstn_img = np.mean(np.load(f'{scores_path}/reduced_norm_first{n+1}_img_ablate_scores.npy'))
        firstn_attn = np.mean(np.load(f'{scores_path}/reduced_norm_first{n+1}_img_out_img2txt_attn_ablate_scores.npy'))

        labels += (f'{n+1}',)
        score_means['Img Toks'] += ((firstn_img/gt_mean)*100,)
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

    ax.set_ylabel('Percent of GT CLIPScore (26.18)')
    ax.set_xlabel('Ablate First N Layers')
    ax.set_title('Ablation Effects on Txt-Normed Image Tokens\nTop-16 Logit CLIPScore')
    ax.set_xticks(x+width, labels)
    ax.axhline(y=(intact_mean / gt_mean)*100, color='b', label='Intact (img norm)')
    ax.axhline(y=(intact_normed_mean / gt_mean)*100, color='g', label='Intact (txt norm)')
    ax.axhline(y=(shuffled_mean / gt_mean)*100, color='r', label='Shuffled (img norm)')
    ax.legend(loc='upper right', ncols=3)
    ax.set_ylim([40, 85])
    plt.show()