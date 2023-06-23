#%%
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from get_class_ratio import get_class_ratio


CLASSES = [
    'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
    'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
    'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
    'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
    'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
    'Triquetrum', 'Pisiform', 'Radius', 'Ulna'
    ]

CLASSES_WITHOUT_ARMBONES = [
    'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
    'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
    'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
    'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
    'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
    'Triquetrum', 'Pisiform'
    ]


def meta_data_EDA():

    json_root = "../data/train/resized_outputs_json"
    meta_root = "../data/meta_data.xlsx"

    # get meta data
    df_meta = pd.read_excel(meta_root)
    # print(df_meta.head(10))
    # print(df_meta.columns)
    df_meta = df_meta.drop('Unnamed: 5', axis=1) # remove Unamed column
    df_meta.rename(columns={'나이':'age', '성별':'gender', '체중(몸무게)':'weight', '키(신장)':'height'}, inplace = True) # change column names
    df_meta.ID = df_meta.ID.apply(str).str.zfill(3).map(lambda x:'ID'+x) # format ID
    df_meta.gender = df_meta.gender.str[-1].replace({'남':'male', '여':'female'}) # format gender
    df_meta = df_meta.set_index(keys='ID') # index = ID
    print("\n## df_meta")
    print(df_meta.head(10))
    print(df_meta.columns)
    print(len(df_meta))
    # datas = df_meta.to_dict('index')
    # print(datas)

    # get class json
    points_count_per_image, points_ratio_per_image, armbone_ratio_per_image = get_class_ratio(json_root)

    df_points_ratio = pd.DataFrame.from_dict(data=points_ratio_per_image, orient='index')
    # print(df_points_ratio.head(10))
    df_points_ratio.insert(0, "image", df_points_ratio.index.map(lambda x: str(x.split('/')[-1].replace('.json', ''))))
    df_points_ratio.index = df_points_ratio.index.map(lambda x: str(x.split('/')[0]))
    print("\n## df_points_ratio")
    print(df_points_ratio.head(10))
    print(df_points_ratio.columns)
    print(len(df_points_ratio))
    # points_ratio_per_image
    # for image_path, datas in points_ratio_per_image.items():
    #     # image_path = 'ID548/image1667354140846.json'
    #     # datas = {'finger-1': 00, 'finger-2': 00, ...}
    #     image_ID = image_path.split('/')[0]
    #     image_name = image_path.split('/')[1]
    #     print(image_ID, image_name)
    #     print(image_name)
    #     break


    # ratio of classes
    
    # plot_points_ratio = df_points_ratio.boxplot(column = classes, rot=45, fontsize=8)
    plot_points_ratio = sns.boxplot(data=df_points_ratio, palette='Set2', order=CLASSES, linewidth=0.8)
    plot_points_ratio.set_xticklabels(CLASSES, size = 8, rotation=45)
    plt.show()
    
    plot_points_ratio_without_armbones = sns.boxplot(data=df_points_ratio, palette='Set2', order=CLASSES_WITHOUT_ARMBONES, linewidth=0.8)
    plot_points_ratio_without_armbones.set_xticklabels(CLASSES_WITHOUT_ARMBONES, size = 8, rotation=45)
    plt.show()


    # merge meta_Data and points_ratio table
    merged_df_points_ratio = pd.merge(df_meta, df_points_ratio, left_index=True, right_index=True)
    print("\n## merged_df_points_ratio")
    print(merged_df_points_ratio.head(10))
    print(merged_df_points_ratio.columns)
    print(len(merged_df_points_ratio))

    # count number of each categories
    # merged_df_points_ratio.sort_values(by='age').age.value_counts(sort=False).plot(kind='bar', fontsize=8)
    # plt.title("age count")
    # plt.ylabel("count")
    # plt.xlabel("age")
    # plt.show()
    create_bar_plot(merged_df_points_ratio.sort_values(by='gender').gender.value_counts(sort=False), title='gender count', xlabel='gender')
    create_bar_plot(merged_df_points_ratio.sort_values(by='age').age.value_counts(sort=False), title='age count', xlabel='age')
    create_bar_plot(merged_df_points_ratio.sort_values(by='weight').weight.value_counts(sort=False), title='weight count', xlabel='weight')
    create_bar_plot(merged_df_points_ratio.sort_values(by='height').height.value_counts(sort=False), title='height count', xlabel='height')

    
    # gender - points_ratio
    # fig, axs_by_gender = plt.subplots(1, 2, figsize=(24, 12))
    # for index, (gender, group) in enumerate(merged_df_points_ratio.groupby('gender')):
    #     print(index)
    #     ax = axs_by_gender[index] # index // 2, index % 2
    #     print(gender)
    #     print(group) # group.T
    #     # plot_points_ratio_by_gender = group.boxplot(column = classes, rot=45, fontsize=8)
    #     ax = sns.boxplot(data=group, palette='Set2', order=classes, ax=ax)
    #     ax.tick_params(labelsize=5, rotation=45)
    #     ax.set(xlabel=gender, ylabel='class')
    # plt.show()
    # fig, axs_by_gender = plt.subplots(6, 5, figsize=(20, 30))
    # for index, c in enumerate(classes):
    #     df = merged_df_points_ratio[['gender', c]]
    #     # print(df)
    #     # print(type(df))
    #     ax = axs_by_gender[index // 5, index % 5] # index // 2, index % 2
    #     sns.boxplot(data=df, y=c, x='gender', palette='Set2', ax=ax)
    #     sns.stripplot(data=df, y=c, x='gender', s = 3, jitter=True, ax=ax)
    #     # sns.swarmplot(data=df, y=c, x='gender', ax=ax)
    #     ax.tick_params(labelsize=5, rotation=45)
    #     ax.set(xlabel='gender', ylabel=c)
    # plt.show()
    create_subplot_by_class(merged_df_points_ratio, category='gender', ax_size=[6,5], fig_size=(20,30))
    create_subplot_by_class(merged_df_points_ratio, category='age', ax_size=[15,2], fig_size=(20,80))
    create_subplot_by_class(merged_df_points_ratio, category='weight', ax_size=[15,2], fig_size=(20,80))
    create_subplot_by_class(merged_df_points_ratio, category='height', ax_size=[15,2], fig_size=(20,80))


def create_subplot_by_class(df, category, ax_size, fig_size):
    fig, axs = plt.subplots(ax_size[0], ax_size[1], figsize=fig_size)
    for index, c in enumerate(CLASSES):
        df_by_class = df[[category, c]]
        # print(df_by_class )
        # print(type(df_by_class ))
        ax = axs[index // ax_size[1], index % ax_size[1]] # index
        sns.boxplot(data=df_by_class, y=c, x=category, palette='Set2', ax=ax)
        sns.stripplot(data=df_by_class, y=c, x=category, s = 2, jitter=True, ax=ax)
        # sns.swarmplot(data=df_by_class, y=c, x=category, ax=ax)
        ax.tick_params(labelsize=5, rotation=45)
        ax.set(xlabel=category, ylabel=c)
    plt.show()


def create_bar_plot(df, title, xlabel, ylabel='count'):
    df.plot(kind='bar', fontsize=8)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


if __name__ == "__main__":
    meta_data_EDA()


# IMAGE Examples
# (기울어진 손) ID276, ID278, ID319, ID289
# (팔뼈 min) ID059/image1661393300595.png / (팔뼈 max) ID468/image1666659890125.png
# %%
