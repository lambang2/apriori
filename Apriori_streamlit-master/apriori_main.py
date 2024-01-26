
import pandas as pd
import streamlit as st
import os


def apriori(data, minSup, minConf=0.7):

    #! ##############
    L = []
    Ldict = []
    result = find_frequent_one_itemsets(data, minSup)
    L1 = result["firstItemSet"]
    # Include the counts from the first itemset
    all_counts = [result["counts"]]
    L.append(list(L1.keys()))
    Ldict.append(L1)

    C1 = pd.DataFrame([result["counts"].keys(), result["counts"].values()]).T.rename(
        columns={0: 'ItemSet', 1: 'Sup-count'})
    col1, col2 = st.columns(2)
    with col1:
        st.write(f'# C{1}')
        st.dataframe(C1)
    with col2:
        st.write(f'# L{1}')
        st.write(pd.DataFrame(
            [L1.keys(), L1.values()]).T.rename(
            columns={0: 'ItemSet', 1: 'Sup-count'}))

    k = 2

    while len(L[k - 2]) > 0:
        before_pruning_Ck, after_pruning_Ck = apriori_gen(L[k - 2])
        counts = count_itemsets(after_pruning_Ck, data)
        Lk = filter_by_min_sup(counts, minSup)
        Ldict.append(Lk)
        L.append(list(Lk.keys()))
        Ck = pd.DataFrame(
            [counts.keys(), counts.values()]).T.rename(
            columns={0: 'ItemSet', 1: 'Sup-count'})
        col1, col2 = st.columns(2)

        with col1:
            st.write(f'# C{k}')
            st.write(Ck)
        with col2:
            st.write(f'# L{k}')
            st.write(pd.DataFrame(
                [Lk.keys(), Lk.values()]).T.rename(
                columns={0: 'ItemSet', 1: 'Sup-count'}))
        all_counts.append(counts)
        k += 1
    if len(Ldict) > 1:
        st.write(f'### So,the Frequent item sets are in L{L.index(L[-2])+1} ')
        st.write(Ldict[-2])

        all_counts_after_pruning = list2Dict(all_counts)
        # st.write(f'# all_counts_after_pruning')
        # st.write(len(all_counts_after_pruning))

        # all_counts_before_pruning = list2Dict(all_counts_before_pruning)
        # st.write(f'# all_counts_before_pruning')
        # st.write(len(all_counts_before_pruning))
        rules = []
        for itemset in Ldict[-2]:
            # st.write(f'---------------')
            # st.write(f'### {itemset}')
            subsets = []
            itemsetList = list(itemset.split(','))
            # st.write(itemsetList)
            for i in range(1, len(itemsetList)):
                itemList = get_subsets(itemsetList, len(itemsetList)-i)
                for set1 in itemList:
                    set1 = ','.join(map(str, set1))
                    subsets.append(set1)

            # st.write("###### subsets")
            # st.write(subsets)

            for i, Set in zip(range(1, len(subsets)+1), subsets):
                # st.write(f'---')
                # st.write(f'#### Rule{i} of ({itemset}) ')
                # st.write(f'confidant = support({itemset}) / support({Set})')
                # if all_counts_after_pruning[itemset] < all_counts_after_pruning[Set]:
                conf = all_counts_after_pruning[itemset] / \
                    all_counts_after_pruning[Set]
                # else:
                #     conf = all_counts_after_pruning[Set] / \
                #         all_counts_after_pruning[itemset]

                rules.append(
                    [itemset, Set, conf, "accepted" if conf >= minConf else "not accepted"])
                # st.write(
                #     f'confidant = support({all_counts_after_pruning[itemset]}) / support({all_counts_after_pruning[Set]}) = {conf}')
                # st.write(
                #     f'##### {"accepted" if conf>=minConf else "not accepted"}')
        dfRules = pd.DataFrame(
            rules, columns=['itemset', 'subset', 'confidence', 'accepted'])
        st.write(f"# Discovering ARs From L{L.index(L[-2])+1}")
        st.table(dfRules)


def list2Dict(list1):
    dict1 = {}
    for dict in list1:
        for key, item in dict.items():
            dict1.update({key: item})
    return dict1


def find_frequent_one_itemsets(D, minSup):
    counts = {}

    for t in D:
        for item in t:
            item = item.rstrip()
            if str(item) == 'nan':
                continue
            counts[item] = counts.get(item, 0) + 1

    first_item_set = {item: counts[item]
                      for item in counts.keys() if counts[item] >= minSup}
    return {"firstItemSet": first_item_set, "counts": counts}


def apriori_gen(Lk_1):
    new_itemsets_before_pruning = []
    new_itemsets_after_pruning = []
    len_Lk_1 = len(Lk_1)

    for i in range(len_Lk_1):
        for j in range(i + 1, len_Lk_1):
            itemset1 = str(Lk_1[i]).split(',')
            itemset2 = str(Lk_1[j]).split(',')

            # print(itemset1[:-1])
            # print(itemset2[:-1])

            if itemset1[:-1] != itemset2[:-1] and itemset1[-1] == itemset2[-1]:
                continue

            new_itemset = sorted(itemset1 + [str(itemset2[-1])])

            if new_itemset in new_itemsets_before_pruning:
                continue
            new_itemsets_before_pruning.append(new_itemset)

            subsets = get_subsets(new_itemset, len(new_itemset) - 1)

            all_subsets_frequent = all(
                ','.join(map(str, subset)) in Lk_1 for subset in subsets
            )

            if all_subsets_frequent:
                new_itemsets_after_pruning.append(
                    ','.join(map(str, new_itemset)))

    return new_itemsets_before_pruning, new_itemsets_after_pruning


def get_subsets(itemset, k):
    if k == 0:
        return [[]]

    if len(itemset) < k:
        return []

    head, *tail = itemset

    subsets_without_head = get_subsets(tail, k)
    subsets_with_head = [(head, *subset)
                         for subset in get_subsets(tail, k - 1)]

    return subsets_without_head + subsets_with_head


def count_itemsets(Ck, D):
    counts = {}
    for c in Ck:
        counts[c] = 0

    for t in D:
        for c in Ck:
            items = c.split(',')
            if all(item in t for item in items):
                counts[c] = counts.get(c, 0) + 1

    return counts


def filter_by_min_sup(counts, minSup):
    return {itemset: counts[itemset] for itemset in counts.keys()
            if counts[itemset] >= minSup}


def prepare_data(D):
    data = []
    for r in D:
        row = []
        for item in r:
            item = item.strip()
            if item != "":
                row.append(item.rstrip())
        data.append(row)

    return data

# lines = []
# with open("DataSet/fromSlide.csv", "r") as f:
#     for line in f.readlines():
#         lines.append(line.split(','))
#         # if line.strip("\n") != delLine:

# result = apriori(lines, 2)


#################### ? UI start here ################################

datasetLocation = {
    "fromSlide.csv": 'DataSet/fromSlide.csv',
    "retail_dataset.csv": 'DataSet/retail_dataset.csv',
    "tesco.csv": 'DataSet/tesco.csv',
}

selected = st.selectbox(label='Pick dataset', options=datasetLocation.keys())

file = st.file_uploader('pick datasets',
                        type=['csv'],
                        label_visibility='collapsed',
                        help='')

# st.write(file.readlines())
# st.write(dir(file))


lines = []
if file is None:
    with open(datasetLocation[selected], "r") as f:
        for line in f.readlines():
            lines.append(line.split(','))
else:
    try:
        os.mkdir('DataSet')
    except OSError:
        pass

    # with open("DataSet/"+file.name, "wr") as f:
    for line in file.readlines():
        lines.append(line.decode('utf-8').split(','))
    # if line.strip("\n") != delLine:


minSup = st.number_input('min Support', value=2)
minConf = st.slider('min confidant', min_value=0.1, max_value=0.99, value=0.7)
data = prepare_data(lines)
st.write(len(data))

if st.button('start'):

    # csv_file = pd.read_csv("DataSet/"+file.name,
    #                        header=None, error_bad_lines=False)

    # st.write('###############')
    # st.write(csv_file.values.tolist())
    st.table(data[0:20])

    apriori(data, minSup, minConf)
