import string
import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F
from llm import correct_ocr_text

from langrecandcrop import ocr_and_crop
from langrecandcrop import crop
from langrecandcrop import ind_lang
from paddlecrop import padddlecrop


from utils import CTCLabelConverter, AttnLabelConverter
from dataset import RawDataset, AlignCollate
from model import Model
import os
import shutil
import easyocr

def organize_images_by_lang(lang_list, base_folder="images"):
    # Make temp folders
    images = sorted(os.listdir(base_folder))
    temp_folders = {}
    for lang in set(lang_list):
        folder = f"temp_{lang}"
        os.makedirs(folder, exist_ok=True)
        temp_folders[lang] = folder

    # Distribute images into temp folders
    for img, lang in zip(images, lang_list):
        src = os.path.join(base_folder, img)
        dst = os.path.join(temp_folders[lang], img)
        shutil.copy(src, dst)

    return temp_folders
def merge_preds(lang_list,preds_dict):
    """
    lang_list: list of language tags (e.g. ["hi","en","hi","en"])
    preds_dict: dict of language -> list of predictions
                all lists have the same length as lang_list
    Returns: merged list
    """
    final = []

    for i, lang in enumerate(lang_list):
        final.append(preds_dict[lang][i])
    return final
def merge_preds(lang_list, preds_dict):
    """
    lang_list: list of language tags (e.g. ["hi","en","hi","en"])
    preds_dict: dict of language -> list of predictions
                all lists have the same length as lang_list
    Returns: merged list
    """
    final = []
    for i, lang in enumerate(lang_list):
        final.append(preds_dict[lang][i])
    return final


# Example usage


def clear_folder(folder_path):
    if os.path.exists(folder_path):
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.remove(file_path)  # remove file or symlink
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # remove subdir
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')
    else:
        print(f"{folder_path} does not exist.")

def print_words_in_image_order(words, boxes, y_threshold=15):
    """
    words: list of recognized words
    boxes: list of bounding boxes (each [[x1,y1],[x2,y2],[x3,y3],[x4,y4]])
    y_threshold: how close vertically two words must be to belong to the same line
    """
    items = []
    for word, box in zip(words, boxes):
        x_coords = [int(p[0]) for p in box]
        y_coords = [int(p[1]) for p in box]
        x_min, y_min = min(x_coords), min(y_coords)
        items.append((x_min, y_min, word))

    # sort by Y first, then X
    items.sort(key=lambda w: (w[1], w[0]))

    # group into lines
    lines = []
    current_line = []
    last_y = None

    for x, y, word in items:
        if last_y is None or abs(y - last_y) <= y_threshold:
            current_line.append((x, word))
            last_y = y if last_y is None else (last_y + y) // 2
        else:
            # finish current line
            current_line.sort(key=lambda w: w[0])
            lines.append(current_line)
            current_line = [(x, word)]
            last_y = y

    if current_line:
        current_line.sort(key=lambda w: w[0])
        lines.append(current_line)

    # print in order

    text = "\n".join(" ".join(word for _, word in line) for line in lines)
        
    return text
    

def make_image_folder(file_path, output_dir="temp_image_folder"):
    """
    """
    # Ensure the file exists
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # Create (or clear) the output directory
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # Copy the image into the folder
    file_name = os.path.basename(file_path)
    dest_path = os.path.join(output_dir, file_name)
    shutil.copy(file_path, dest_path)

    return output_dir


# Example usage:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def demo(opt):
    """ model configuration """
    if 'CTC' in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)

    if opt.rgb:
        opt.input_channel = 3
    model = Model(opt)

    model = torch.nn.DataParallel(model).to(device)

    # load model
    print('loading pretrained model from %s' % opt.saved_model)
    model.load_state_dict(torch.load(opt.saved_model, map_location=device))

    # prepare data. two demo images from https://github.com/bgshih/crnn#run-demo
    AlignCollate_demo = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
    demo_data = RawDataset(root=opt.image_folder, opt=opt)  # use RawDataset
    demo_loader = torch.utils.data.DataLoader(
        demo_data, batch_size=opt.batch_size,
        shuffle=False,
        num_workers=int(opt.workers),
        collate_fn=AlignCollate_demo, pin_memory=True)

    # predict
    model.eval()
    with torch.no_grad():
        for image_tensors, image_path_list in demo_loader:
            batch_size = image_tensors.size(0)
            image = image_tensors.to(device)
            # For max length prediction
            length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
            text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)

            if 'CTC' in opt.Prediction:
                preds = model(image, text_for_pred, is_train=False)

                # Select max probabilty (greedy decoding) then decode index to character
                preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                _, preds_index = preds.max(2)
                # preds_index = preds_index.view(-1)
                preds_str = converter.decode(preds_index, preds_size)

            else:
                preds = model(image, text_for_pred, is_train=False)

                # select max probabilty (greedy decoding) then decode index to character
                _, preds_index = preds.max(2)
                preds_str = converter.decode(preds_index, length_for_pred)




            preds_prob = F.softmax(preds, dim=2)
            preds_max_prob, _ = preds_prob.max(dim=2)

            

            # for img_name, pred, pred_max_prob in zip(image_path_list, preds_str, preds_max_prob):
            #     if 'Attn' in opt.Prediction:
            #         pred_EOS = pred.find('[s]')
            #         pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
            #         pred_max_prob = pred_max_prob[:pred_EOS]

            #     # calculate confidence score (= multiply of pred_max_prob)
            #     confidence_score = pred_max_prob.cumprod(dim=0)[-1]

            #     print(f'{img_name:25s}\t{pred:25s}\t{confidence_score:0.4f}\t{lang}')
                # log.write(f'{img_name:25s}\t{pred:25s}\t{confidence_score:0.4f}\n')

   
            return preds_str

def text_recognition(img_path,y):

    parser = argparse.ArgumentParser()
    parser.add_argument('--image_folder', help='path to image_folder which contains text images')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--batch_size', type=int, default=192, help='input batch size')
    parser.add_argument('--saved_model', help="path to saved_model to evaluation")
    """ Data processing """
    parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
    parser.add_argument('--imgH', type=int, default=64, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=200, help='the width of the input image')
    parser.add_argument('--rgb', action='store_true', help='use rgb input')
    parser.add_argument('--character', type=str, default="0123456789!'#$%&\"()*+,-./:;<=>?@[\\]^_`{|}~ abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ।ঁংঃঅআইঈউঊঋঌএঐওঔকখগঘঙচছজঝঞটঠডঢণতথদধনপফবভমযরলশষসহ়ািীুূৃেৈোৌ্ৎড়ঢ়য়০১২৩৪৫৬৭৮৯", help='character label')
    parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
    parser.add_argument('--PAD', action='store_true', default=True, help='whether to keep ratio then pad for image resize')
    """ Model Architecture """
    parser.add_argument('--Transformation', type=str , help='Transformation stage. None|TPS')
    parser.add_argument('--FeatureExtraction', type=str, help='FeatureExtraction stage. VGG|RCNN|ResNet')
    parser.add_argument('--SequenceModeling', type=str, help='SequenceModeling stage. None|BiLSTM')
    parser.add_argument('--Prediction', type=str , help='Prediction stage. CTC|Attn')
    parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
    parser.add_argument('--input_channel', type=int, default=1, help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=512,
                        help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=512, help='the size of the LSTM hidden state')

    opt = parser.parse_args()
    opt.Transformation = 'None'
    opt.FeatureExtraction = 'ResNet'
    opt.SequenceModeling = 'BiLSTM' 
    opt.Prediction = 'CTC'
    """ vocab / character number configuration """
    # opt.image_folder = image_folder1
    if y in  ['en','de', 'es', 'fr', 'it', 'nl', 'pl', 'pt', 'sw', 'tr', 'vi', 'zh','hi', 'mr','ha_hi','ar']:

        lang = y
        if lang in ['en','de', 'es', 'fr', 'it', 'nl', 'pl', 'pt', 'sw', 'tr', 'vi']:
            lang1 = 'en'
        elif lang == 'zh':
            lang1 = 'ch_tra'
        elif lang == 'hi' or lang == 'mr':
            lang1 = 'hi'
        if lang == 'ar':
            lang1 = 'ar'

        
        if lang == 'ha_hi':
            opt.saved_model = r"D:\OCR\saved_models\hindi_handwritten.pth"
            opt.character = '0123456789!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.ँंःअअंअःआइईउऊऋएऐऑओऔकखगघङचछजझञटठडढणतथदधनऩपफबभमयरऱलळवशषसह़ािीुूृॅेैॉोौ्ॐ॒क़ख़ग़ज़ड़ढ़फ़ॠ।०१२३४५६७८९॰'
            print(f'the langauge that you have chosen is {lang}')
        
            clear_folder(r"D:\OCR\deeptextrecognitionbenchmark\crops")
            coordinates = padddlecrop(img_path)
            lang_list = ['hi'] * len(coordinates)
            opt.image_folder = r"D:\OCR\deeptextrecognitionbenchmark\crops"
            cudnn.benchmark = True
            cudnn.deterministic = True
            opt.num_gpu = torch.cuda.device_count()

            pred = demo(opt)
        else:
            coordinates,pred = ind_lang(img_path,lang1)
            lang_list = [lang] * len(coordinates)


        text = print_words_in_image_order(pred,coordinates,y_threshold=30)
        



    else:
        clear_folder(r"D:\OCR\crops")
        lang_list, bounding_boxes,en_text,hi_text,ch_text,ar_text = ocr_and_crop(img_path)
        
        if "ch_tra" in lang_list:
            # opt.saved_model = r"D:\OCR\saved_models\final_chi.pth"
            # opt.character = "0123456789!#$%&\"'()*+,-./:;<=>?@[\\]^_`{|}~ abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ一丁七丈三上下不丑且丕世丘丙丞丟並丫中丰串丸丹主乃久么之乍乎乏乒乓乖乘乙九乞也乳乾亂了予事二于云互五井些亞亟亡亢交亥亦亨享京亭亮亳亶人什仁仃仄仆仇今介仍仔仕他仗付仙仞仡代令以仰仲件价任份仿企伉伊伍伎伏伐休伕伙伯估伴伶伸伺似伽佃但佈位低住佐佑佔何佗佘余佚佛作佝佞佟你佣佤佩佬佯佰佳併佻佼使侃侄來侈例侍侏侑侖侗供依侮侯侵侶侷便係促俄俊俏俐俑俗俘俚保俞俟俠信俬修俯俱俳俵俸俺俾倀倆倉個倍們倒倔倖倘候倚倜借倡倣倦倨倩倪倫倬倭值偃假偈偉偌偏偕做停健側偵偶偷偽傀傅傌傍傑傘備傚傢傣催傭傲傳債傷傻傾僂僅僉像僑僕僖僚僥僧僭僮僰僱僳僵價僻儀儂億儆儉儋儐儒儘償儡優儲儸儻儼儿兀允元兄充兆兇先光克兌免兒兔兕兗兜入內全兩八公六兮共兵其具典兼冀冉冊再冑冒冕冗冠冢冤冥冬冰冶冷冼准凋凌凍凜凝几凡凰凱凳凶凸凹出函刀刁刃分切刈刊刎刑划列初判別刨利刪刮到制刷券刺刻剁剃則削剌前剎剔剖剛剝剡剩剪副割創剷剽剿劃劇劈劉劍劑力功加劣助努劫劭劾勁勃勇勉勒動勖勗勘務勛勝勞募勢勤勦勰勳勵勸勺勻勾勿包匆匈匍匐匕化北匙匝匠匡匣匪匯匱匹匾匿區十千卅升午卉半卑卒卓協南博卜卞占卡卦卯印危即卵卷卸卹卻卿厄厘厚厝原厥厭厲去參又叉及友反叔取受叛叟叡叢口古句另叨叩只叫召叭叮可台叱史右司叻叼吁吃各合吉吊吋同名后吏吐向吒君吝吞吟吠否吧吩含听吳吵吶吸吹吻吼吾呀呂呃呆呈告呎呢呤周味呵呸呻呼命咀咁咄咆和咎咐咒咕咖咚咧咨咩咪咫咬咭咯咱咳咸咽哀品哄哆哇哈哉員哥哦哨哩哪哭哮哲哺哼哿唁唄唆唇唉唐唑唔唧唬售唯唱唸唾啃啄商啊問啕啞啟啡啣啤啥啦啶啼啾喀喂喃善喇喉喊喋喔喘喙喚喜喝喟喧喪喫喬單喱喻嗅嗎嗓嗔嗜嗟嗡嗣嗦嗩嗯嗲嗶嗽嘆嘈嘉嘌嘎嘔嘖嘗嘛嘧嘩嘮嘯嘲嘴嘸嘻嘿噁噎器噩噪噬噴噶噸噹嚇嚎嚏嚐嚥嚨嚮嚴嚼囂囉囊囍囑囚四回因囤困囹固圃圄圈國圍園圓圖團圜土在圩圭圮地圳圻圾址均坊坌坍坎坏坐坑坡坤坦坨坪坭坳坷坻垂垃型垓垚垛垠垢垣垮埂埃埋城埏埒埔埕域埠埤埭執埸培基埼堀堂堅堆堇堉堊堍堠堡堤堪堯堰報場堵塊塌塑塔塗塘塚塞塢塤填塵塾墀境墅墉墊墓墘墜增墟墨墩墮墳墾壁壅壇壑壓壕壘壙壞壟壢壤壩士壬壯壹壺壼壽夏夔夕外夙多夜夠夢夥大天太夫夭央夯失夷夸夾奄奇奈奉奎奏奐契奔奕套奘奚奠奢奧奪奭奮女奴奶奸她好如妃妄妊妍妒妓妖妙妝妞妣妤妥妨妮妲妳妹妻妾姆姊始姐姑姒姓委姚姜姝姣姥姦姨姪姬姮姻姿威娃娑娘娛娜娟娠娣娥娩娶娼婁婆婉婕婚婢婦婧婪婭婷婺婿媒媚媛媧媲媳媽嫁嫂嫉嫌嫖嫚嫡嫣嫦嫩嫻嬉嬋嬌嬗嬛嬤嬪嬬嬰嬴嬸嬿孀孃孌子孔孕孖字存孚孛孜孝孟孢季孤孩孫孱孵學孺孽孿宁它宅宇守安宋完宏宓宕宗官宙定宛宜客宣室宥宦宮宰害宴宵家宸容宿寂寄寅密寇富寐寒寓寞察寡寢寥實寧寨審寫寬寮寰寵寶寸寺封射將專尉尊尋對導小少尕尖尚尤尪尬就尷尸尹尺尻尼尾尿局屁居屆屈屋屍屎屏屑展屜屠屢層履屬屯山岌岐岑岔岡岩岫岬岱岳岷岸峇峋峒峙峨峪峭峰島峻峽崁崆崇崎崑崔崖崗崙崚崛崞崢崤崧崩崮崴崽嵊嵌嵎嵐嵩嵬嵯嶄嶇嶙嶲嶷嶺嶼嶽巍巒巔巖川州巡巢工左巧巨巫差巰己已巳巴巷巽巾市布帆希帑帕帖帘帚帛帝帥師席帳帶帷常帽幄幅幌幕幗幘幟幡幢幣幫干平年并幸幹幻幼幽幾庄庇床庋序底庖店庚府庠度座庫庭庵庶康庸庹庾廁廂廈廉廊廒廓廕廖廚廝廟廠廡廢廣廩廬廳延廷建廿弁弄弈弊弋式弒弓弔引弗弘弛弟弢弦弧弩弱張強弼彆彈彊彌彎彗彘彙形彤彥彧彩彪彬彭彰影彷役彼彿往征待徇很徊律後徐徑徒得徘徙從徠御徨復循徬徭微徵德徹徽心必忌忍忐忑忒忖志忘忙忠忡忤快忭忱念忸忻忽忿怎怒怕怖思怠怡急性怨怩怪怯恁恂恃恆恉恍恐恕恙恢恣恤恥恨恩恪恫恬恭息恰恿悄悅悉悌悍悔悖悚悝悟悠患您悱悲悵悶悸悼悽情惆惇惊惋惑惕惚惜惟惠惡惦惰惱想惶惹惺惻愁愆愈愉愍意愕愚愛感愧愫愷愿慄慈態慌慎慕慘慚慟慢慣慧慨慫慮慰慵慶慷慾憂憊憎憐憑憚憤憧憨憩憫憬憲憶憾懂懃懇懈應懊懋懣懦懲懶懷懸懺懼懾懿戀戇戈戊戌戍戎成我戒戕或戚戛戟戡截戮戰戲戳戴戶戾房所扁扇扈手才扎扑扒打扔托扛扣扦扭扮扯扳扶批扼找承技抃抄抉把抑抒抓投抖抗折抨披抬抱抵抹押抽拂拄拆拇拈拉拋拌拍拎拐拒拓拔拖拗拘拙拚招拜括拭拮拯拱拳拴拷拼拽拾拿持挂指按挑挖挨挪挫振挹挺挽挾捂捅捆捉捍捎捏捐捕捧捨捩捫据捱捲捶捷捺捻掀掃掄授掉掌掎掏掐排掖掘掙掛掟掠採探掣接控推掩措掰掾揀揆揉揍描提插揖揚換握揣揩揪揭揮援揹損搏搐搓搖搗搜搞搡搥搪搬搭搶摒摔摘摧摩摭摯摶摸摹摺摻摽撂撇撈撐撒撓撕撞撤撥撩撫撬播撮撰撲撻撼撾撿擁擂擄擅擇擊擋操擎擒擔擘據擠擢擦擬擭擱擲擴擺擾攀攄攏攔攘攙攜攝攣攤攪攫攬支收攸改攻放政故效敏救敔敕敖敗敘教敝敞敢散敦敬敲整敵敷數斂斃文斌斐斑斗料斛斜斟斡斤斥斧斫斬斯新斷方於施旁旃旄旅旋旌族旒旗既日旦旨早旬旭旱旺旻昀昂昃昆昇昊昌明昏易昔昕昝星映昤春昧昨昭是昱昴昵昶時晃晉晏晒晙晚晝晞晟晤晦晨普景晰晴晶晸智晾暄暇暈暉暌暐暑暖暗暝暠暢暨暫暮暱暴暹曄曆曉曖曙曜曝曠曦曩曬曰曲曳更書曹曼曾替最會月有朋服朔朕朗望朝期木未末本札朱朴朵朽杆杉李杏材村杖杜杞束杠杭杯杰東杲杳杵杷杻杼松板枇枉枋析枕林枚果枝枯架枷枸柁柄柏某柑染柔柘柚柜柝柞柢查柩柬柯柰柱柳柴柵柿栓栖栗栝校栩株栲核根格栽桀桂桃桅框案桉桌桎桐桑桓桴桶桿梁梅梆梏梓梗條梟梢梧梨梭梯械梳梵棄棉棋棍棐棒棕棗棘棚棟棠棣棧棨棪棫森棱棲棵棹棺棻棼椅椋植椎椏椒椰椴椿楊楓楔楙楚楞楠楢楣楨楫業極楷楹概榆榎榔榕榛榜榨榭榮榴榷榻榿槃槊構槍槎槐槓槙槤槭槳槻槽槿樁樂樊樑樓樗標樞樟模樣樵樸樹樺樽橄橇橈橋橘橙機橡橢橫橿檀檄檎檐檔檗檜檢檬檯檳檸檻櫃櫓櫚櫛櫟櫥櫧櫨櫬櫸櫻欄權欒欖欠次欣欲欹欺欽款歃歆歇歉歌歎歐歙歜歟歡止正此步武歧歪歲歷歸歹死殂殃殆殉殊殖殘殫殭殮殯殲段殷殺殼殿毀毅毆毋母每毒毓比毖毗毘毛毫毬毯毽氏氐民氓气氘氙氚氛氟氣氦氧氨氫氬氮氯氰水永氾汀汁求汊汎汐汕汗汙汛汜汝汞江池污汨汪汰汲汴汶決汽汾沁沂沃沅沈沉沌沐沒沓沔沖沙沚沛沫沭沮沱河沸油治沼沽沾沿況泄泅泉泊泌泓法泖泗泚泛泡波泣泥注泫泮泰泳泵洄洋洌洒洗洙洛洞津洩洪洮洱洲洵洶洸洹活洽派流浙浚浦浩浪浮浴海浸涂涅涇消涉涌涕涪涮涯液涵涼涿淀淄淅淆淇淋淌淑淒淖淘淙淚淞淡淤淦淨淪淫淮淯深淳淵混淹淺添淼清渚減渝渠渡渣渤渥渦測渭港渲渴游渺渾湃湄湊湍湖湘湛湜湟湣湧湮湯溉源準溘溜溝溟溢溥溧溪溫溯溲溴溶溺溼溽滁滂滄滅滇滋滌滎滑滓滔滕滘滬滯滲滴滷滸滾滿漁漂漆漏漓演漕漠漢漣漩漪漫漬漱漲漳漶漸漾漿潁潑潔潘潛潞潟潢潤潭潮潯潰潸潼澀澂澄澆澈澍澎澔澗澡澤澧澮澱澳澹激濁濂濃濉濕濘濛濟濠濡濤濫濬濮濰濱濺濾瀆瀉瀋瀏瀑瀕瀘瀚瀛瀝瀟瀦瀧瀨瀾灌灑灘灝灞灣火灰灶灸灼災炅炊炎炒炕炖炙炤炫炬炭炮炯炳炸為烜烤烯烴烷烹烽焉焊焙焚焜無焦焯焰然煃煇煉煌煎煒煙煜煞煤煥照煩煬煮煲煽熄熊熏熒熔熙熟熨熬熱熵熹熾燁燃燄燈燉燊燎燒燕燙營燥燦燧燬燭燮燹燼燾爆爇爍爐爛爪爬爭爰爵父爸爹爺爽爾牂牆片版牌牒牘牙牛牟牠牡牢牧物牲特牽犀犁犍犒犛犢犧犬犯狀狂狄狌狐狗狙狠狡狩狸狹狼狽猁猖猗猛猜猝猞猥猩猴猶猷猾猿獄獅獎獗獠獨獲獵獷獸獻獼獾玄率玉王玕玖玟玠玥玦玩玫玲玳玷玻珀珂珈珊珍珙珞珠珩珪班珮珽現球琅理琇琉琊琍琚琛琢琥琦琨琪琮琯琰琳琴琵琶瑁瑋瑕瑙瑚瑛瑜瑞瑟瑣瑤瑩瑪瑭瑯瑰瑾璀璃璆璇璉璋璐璘璜璞璟璦璧璨璫環璽璿瓊瓏瓚瓜瓢瓣瓦瓮瓶瓷甄甕甘甚甜生產甥甦用甩甫甬甯田由甲申男甸町甾畀畈畋界畏畔留畜畝畢畤略畦番畫異當畸畿疆疇疊疏疑疙疚疝疣疤疥疫疲疳疵疹疼疾病症痊痍痔痕痘痙痛痞痢痣痰痲痳痴痹痺痿瘀瘁瘉瘋瘍瘓瘟瘠瘡瘤瘦瘧瘩瘳瘴療癆癌癒癘癟癡癢癬癮癱癸登發白百皂的皆皇皈皋皎皓皖皚皮皰皺皿盂盃盅盆盈益盎盒盔盛盜盞盟盡監盤盥盧盪目盯盱盲直相盼盾省眈眉看眙真眠眨眩眭眯眶眷眸眺眼眾睇睛睜睞睡睢督睦睪睫睬睹睽睾睿瞄瞇瞋瞎瞑瞞瞢瞧瞪瞬瞭瞰瞳瞻瞿矗矚矛矜矢矣知矩短矮矯石矻矽砂砆砌砍研砝砟砢砦砧砭砲破砵砸硃硅硒硝硤硨硫硬确硯硼硿碇碌碎碑碓碗碘碚碟碣碧碩碭碰碳碴確碻碼碾磁磅磊磋磐磔磚磡磧磨磬磯磲磷磺礁礎礑礙礦礪礫示礽社祀祁祂祇祈祉祊祐祕祖祗祚祛祜祝神祟祠祥祧票祭祺祿禁禍禎福禕禦禧禪禮禱禳禹禺禽禾禿秀私秉秋科秒秕秘租秣秤秦秧秩秫秸移稀稃稅程稍稔稗稙稚稜稞稟稠種稱稷稻稼稽稿穀穆穌積穎穗穢穩穫穴究穹空穿突窄窆窈窒窕窖窗窘窟窠窨窩窪窮窯窺窿竄竅竇竊立竑站竟章竣童竭端競竹竺竽竿笈笏笑笘笙笛笞笠符笨第笭筅筆等筊筋筍筏筐筑筒答策筠筩筮筱筲筵筷箇箋箍箏箔箕算管箬箭箱箴箸節範篆篇築篙篚篠篡篤篦篩篷篾簇簋簑簡簧簪簷簸簽簾簿籀籃籌籍籙籠籤籬籲米籽粉粑粒粕粗粘粟粥粱粲粵粹粽精粿糊糌糕糖糙糜糞糟糠糧糯糰糴糸系糾紀約紅紆紉紊紋納紐紓純紗紘紙級紛紜素紡索紫紮累細紳紹紺絀終絃組結絕絛絜絞絡絢給絨絮統絲絳絹綁綏綑經綜綠綢綦綬維綱網綴綵綸綺綻綽綾綿緊緋緒線緝緞締緣編緩緬緯練緻縉縊縑縛縝縞縣縫縮縯縱縷總績繁繃繆織繕繖繞繡繩繪繫繭繳繹繼纂纇纈續纏纓纖纘纛纜缶缸缺缽罄罌罐罔罕罘罟罩罪置罰署罵罷罹羅羆羈羊羌美羔羚羞群羥羧羨義羯羰羲羸羹羽羿翁翅翊翌翎習翔翟翠翡翥翦翩翮翰翱老考耄者耆耋而耍耐耒耕耖耗耘耙耜耦耳耶耽耿聆聊聒聖聘聚聞聯聰聲聳聶職聽聾聿肄肅肆肇肉肋肌肖肘肚肛肜肝股肢肥肩肪肫肯肱育肺胂胃胄背胎胖胚胛胞胡胤胥胭胯胰胱胸胺能脂脅脆脈脊脖脛脣脩脫脹脾腆腊腋腌腎腐腑腓腔腕腥腦腧腫腰腱腳腴腸腹腺腿膀膂膈膊膏膚膛膜膝膠膨膩膳膺膽膾膿臀臂臃臆臉臊臘臚臞臟臣臥臧臨自臬臭至致臺臻臼臾舀舂舅與興舉舊舌舍舒舔舖舜舞舟航般舵舶舷舸船舺艇艮良艱色艷艾芊芋芍芎芒芙芝芡芥芩芫芬芭芮芯花芳芷芸芹芽芾苑苒苓苔苕苗苛苜苞苟苡苣若苦苧苯英苳苹苻苾茁茂范茄茅茆茉茌茗茛茜茨茫茯茱茲茴茵茶茸茹荀荃荅草荊荏荐荒荔荖荷荸荻荼荽莆莉莊莎莒莓莖莘莞莢莪莫莽菀菁菅菇菊菌菏菑菘菜菠菡菩華菱菲菴菸萁萃萄萇萊萌萍萎萩萬萱萸萼落葆葉著葛葡董葦葫葬葳葵葶葺蒂蒐蒔蒙蒜蒞蒡蒨蒯蒲蒴蒸蒺蒼蒿蓀蓄蓆蓉蓋蓑蓓蓬蓮蓼蔑蔓蔗蔚蔡蔣蔥蔬蔭蔻蔽蕃蕈蕉蕊蕙蕞蕨蕩蕪蕭蕾薄薇薈薊薑薙薛薜薦薨薩薪薯薰薹薺藉藍藏藐藕藜藝藤藥藨藩藪藻蘄蘅蘆蘇蘊蘋蘑蘚蘧蘭蘸蘼蘿虎虐虔處虖虛虜虞號虧虫虱虹蚊蚌蚓蚕蚜蚣蚤蚨蚩蚪蚵蚺蛄蛆蛇蛉蛋蛐蛔蛙蛛蛟蛤蛩蛭蛹蛺蛻蛾蜀蜂蜆蜈蜊蜑蜒蜓蜘蜚蜜蜡蜥蜱蜴蜷蜻蜿蝌蝕蝗蝘蝙蝠蝦蝮蝴蝶蝸螂螃螈融螞螟螢螭螯螳螺螽蟀蟄蟆蟈蟋蟑蟠蟬蟲蟹蟻蟾蠃蠅蠆蠍蠑蠓蠔蠕蠟蠡蠢蠣蠱蠲蠶蠻血行衍術街衙衛衝衡衢衣表衫衰衷衹袁袋袍袒袖袛袞袪被袱裁裂裔裕裘裙補裝裡裨裱裲裳裴裸裹製裾褂複褐褒褚褡褥褪褫褲褶褸褻襄襖襟襠襤襦襪襬襯襲西要覃覆見規覓視覘親覲覺覽觀角觔觚觝解觴觸觿言訂訃訇計訊訌討訐訓訕訖託記訛訝訟訢訣訥訪設許訴訶診註証詐詒詔評詗詛詞詠詢試詩詫詬詭詮詰話該詳詹詼誅誇誌認誓誕誘語誠誡誣誤誥誦誨說誰課誼調諂談請諍諏諒論諛諜諡諤諦諧諫諭諮諱諳諷諸諺諾謀謁謂謄謇謊謎謗謙謚講謝謠謨謫謬謳謹謾譁證譏識譙譚譜警譬譯議譴護譽讀變讓讖讙讚讞谷谿豁豆豈豉豊豌豎豐豔豚象豢豹豺貂貉貊貌貍貓貝貞負財貢貧貨販貪貫責貴貶買貸費貼貽貿賀賁賂賃賄資賈賊賑賓賚賜賞賠賡賢賣賤賦質賬賭賴賸賺購賽贇贈贊贏贓贖贛赤赦赫赭走赴赶起趁超越趕趙趟趣趨足趴趵趾跆跋跌跑跖跗跛距跟跡跣跤跨跪路跳踊踏踐踝踞踢踩踰踴踵踹蹂蹄蹇蹈蹊蹋蹔蹟蹠蹤蹦蹬蹭蹯蹲蹴蹶蹼躁身躬躲躺軀車軋軌軍軒軔軛軟軫軸軾較輅載輒輓輔輕輛輜輝輟輩輪輯輸輻輾輿轂轄轅轉轍轎轟辛辜辟辣辦辨辭辯辰辱農迂迄迅迎近返迢迤迥迦迨迪迫迭述迴迷迺追退送适逃逄逅逆逋逍透逐逑途逕逗這通逛逝逞速造逢連逮逯週進逵逶逸逼逾遁遂遇遊運遍過遏遐遑遒道達違遘遙遜遞遠遣遨適遭遮遲遴遵遶遷選遺遼遽避邀邁邂還邇邈邊邏邑邕邛邠邢那邦邪邯邰邱邳邴邵邸邽邾郁郃郅郇郊郎郛郜郝郡郢部郭郯郴郵都郾鄂鄉鄒鄔鄙鄞鄧鄭鄯鄰鄱鄴鄺酆酉酊酋酌配酐酒酗酚酣酥酩酪酬酮酯酴酵酷酸醅醇醉醋醍醐醒醚醛醜醞醣醫醬醮醯醴釀釁采釉釋里重野量釐金釗釘釜針釣釦釧釵釷鈉鈍鈔鈕鈞鈣鈴鈷鈸鈺鈾鉀鉅鉉鉑鉗鉚鉛鉞鉤鉬鉸鉻銀銃銅銓銖銘銜銥銨銫銳銷銼鋁鋅鋆鋒鋤鋪鋰鋸鋼錄錐錒錕錘錚錠錡錢錦錨錫錮錯錳錶鍊鍋鍍鍛鍥鍬鍰鍱鍵鍶鍼鍾鎂鎊鎌鎖鎗鎚鎡鎧鎩鎬鎮鎳鏈鏖鏗鏘鏞鏟鏡鏢鏤鏽鐘鐙鐮鐵鐸鑄鑊鑑鑒鑣鑫鑰鑱鑲鑷鑼鑽鑾鑿長門閂閃閉開閏閑閒間閔閘閡閣閤閥閨閩閫閭閱閹閻閼閾闆闇闈闊闌闓闔闕闖關闞闡闢阜阡阪阮阱防阻阽阿陀陂附陋陌降限陛陝陞陟陡院陣除陪陰陲陳陵陶陷陸陽隄隅隆隈隊隋隍階隔隕隗隘隙際障隧隨險隱隴隸隹隻隼雀雁雄雅集雇雉雋雌雍雎雒雕雖雘雙雛雜雞離難雨雪雯雰雲零雷雹電需霄霆震霉霍霎霏霑霓霖霙霜霞霧霨霰露霸霹霽霾靂靄靈青靖靚靛靜非靠靡面靨革靳靴靶靼鞅鞋鞍鞏鞘鞠鞣鞭韁韃韋韌韓韜韞韭音韶韻響頁頂頃項順須頊頌頏預頑頒頓頗領頜頡頤頦頭頰頷頸頹頻顆題額顎顏顒顓願顙顛類顥顧顫顯顱風颯颱颶飄飆飛食飢飩飪飭飯飲飴飼飽飾餅餉養餌餐餒餓餘餚餛餞餡館餬餵餽餾饅饈饋饌饑饒饕饗首馗香馥馨馬馭馮馳馴駁駐駒駕駙駛駝駭駱駿騁騎騏騑騙騧騫騰騷騾驁驃驄驅驊驍驕驗驚驛驟驢驤驩驪骨骰骷骸骼髀髏髒髓體髖高髦髮髯髻鬃鬆鬍鬘鬚鬟鬣鬥鬧鬩鬯鬱鬲鬻鬼魁魂魄魅魋魏魔魚魨魯魴魷鮆鮑鮚鮠鮨鮪鮫鮭鮮鮸鯁鯉鯊鯔鯖鯛鯡鯧鯨鯰鯽鰈鰍鰓鰨鰭鰲鰻鱄鱈鱔鱗鱘鱠鱧鱨鱷鱸鱺鳥鳩鳳鳴鳶鴉鴒鴛鴞鴦鴨鴻鴿鵝鵠鵡鵪鵬鵰鵲鶉鶩鶯鶴鶿鷗鷳鷸鷹鷺鸕鸚鸛鸞鹵鹹鹼鹽鹿麂麇麋麒麓麗麝麟麥麩麴麵麻麼麾黃黎黏黑黔默黛黜點黠黧黨黯黴黷黼鼎鼐鼓鼠鼬鼻齊齋齒齡齣齦齧龍龐龔龕龜龢"
            # opt.image_folder =  r"D:\OCR\crops"
            # cudnn.benchmark = True
            # cudnn.deterministic = True
            # opt.num_gpu = torch.cuda.device_count()

            pred_ch = ch_text
        if "en" in lang_list:
            # opt.saved_model = r"D:\OCR\saved_models\en_filtered.pth"
            # opt.character = "0123456789!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzÀÁÂÃÄÅÆÇÈÉÊËÍÎÑÒÓÔÕÖØÚÛÜÝÞßàáâãäåæçèéêëìíîïðñòóôõöøùúûüýþÿąęĮįıŁłŒœŠšųŽž"
            # opt.image_folder =  r"D:\OCR\crops"
            # cudnn.benchmark = True
            # cudnn.deterministic = True
            # opt.num_gpu = torch.cuda.device_count()


            pred_en = en_text

        if "hi" in lang_list:
            # opt.saved_model = r"D:\OCR\saved_models\best_accuracy_hi.pth"
            # opt.character = '0123456789!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.ँंःअअंअःआइईउऊऋएऐऑओऔकखगघङचछजझञटठडढणतथदधनऩपफबभमयरऱलळवशषसह़ािीुूृॅेैॉोौ्ॐ॒क़ख़ग़ज़ड़ढ़फ़ॠ।०१२३४५६७८९॰'
            # opt.image_folder =  r"D:\OCR\crops"
            # cudnn.benchmark = True
            # cudnn.deterministic = True
            # opt.num_gpu = torch.cuda.device_count()

            # pred_hi = demo(opt) 
            pred_hi = hi_text
        if "ar" in lang_list:
            # opt.saved_model = r"D:\OCR\crops\best_accuracy_ar.pth"
            # opt.character = '0123456789!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ اابتثجحخدذرزسشصضطظعغفقكلمنهويىآأؤإئًٌٍَُِّْ٠١٢٣٤٥٦٧٨٩٪٫٬؛؟'
            # opt.image_folder =  r"D:\OCR\crops"
            # cudnn.benchmark = True
            # cudnn.deterministic = True
            # opt.num_gpu = torch.cuda.device_count()

            pred_ar = ar_text

        preds_dict = {}
        if 'hi' in lang_list:
            preds_dict['hi'] = pred_hi
        if 'en' in lang_list:
            preds_dict['en'] = pred_en
        if 'ch_tra' in lang_list:
            preds_dict['ch_tra'] = pred_ch
        if 'ar' in lang_list:
            preds_dict['ar'] = pred_ar
        pred = merge_preds(lang_list, preds_dict)
        text = print_words_in_image_order(pred,bounding_boxes)

        

    return bounding_boxes,lang_list,pred,text


            


        
        



            



        

    # else:

        # print(f"Detected language: {lang}")

#     if lang in  ['en','de', 'es', 'fr', 'it', 'nl', 'pl', 'pt', 'sw', 'tr', 'vi']:
#         opt.saved_model = r"D:\OCR\saved_models\en_filtered.pth"
#         opt.character = "0123456789!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzÀÁÂÃÄÅÆÇÈÉÊËÍÎÑÒÓÔÕÖØÚÛÜÝÞßàáâãäåæçèéêëìíîïðñòóôõöøùúûüýþÿąęĮįıŁłŒœŠšųŽž"
#     elif lang == 'zh':
#         opt.saved_model = r"D:\OCR\saved_models\final_chi.pth"
#         opt.character = "0123456789!#$%&\"'()*+,-./:;<=>?@[\\]^_`{|}~ abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ一丁七丈三上下不丑且丕世丘丙丞丟並丫中丰串丸丹主乃久么之乍乎乏乒乓乖乘乙九乞也乳乾亂了予事二于云互五井些亞亟亡亢交亥亦亨享京亭亮亳亶人什仁仃仄仆仇今介仍仔仕他仗付仙仞仡代令以仰仲件价任份仿企伉伊伍伎伏伐休伕伙伯估伴伶伸伺似伽佃但佈位低住佐佑佔何佗佘余佚佛作佝佞佟你佣佤佩佬佯佰佳併佻佼使侃侄來侈例侍侏侑侖侗供依侮侯侵侶侷便係促俄俊俏俐俑俗俘俚保俞俟俠信俬修俯俱俳俵俸俺俾倀倆倉個倍們倒倔倖倘候倚倜借倡倣倦倨倩倪倫倬倭值偃假偈偉偌偏偕做停健側偵偶偷偽傀傅傌傍傑傘備傚傢傣催傭傲傳債傷傻傾僂僅僉像僑僕僖僚僥僧僭僮僰僱僳僵價僻儀儂億儆儉儋儐儒儘償儡優儲儸儻儼儿兀允元兄充兆兇先光克兌免兒兔兕兗兜入內全兩八公六兮共兵其具典兼冀冉冊再冑冒冕冗冠冢冤冥冬冰冶冷冼准凋凌凍凜凝几凡凰凱凳凶凸凹出函刀刁刃分切刈刊刎刑划列初判別刨利刪刮到制刷券刺刻剁剃則削剌前剎剔剖剛剝剡剩剪副割創剷剽剿劃劇劈劉劍劑力功加劣助努劫劭劾勁勃勇勉勒動勖勗勘務勛勝勞募勢勤勦勰勳勵勸勺勻勾勿包匆匈匍匐匕化北匙匝匠匡匣匪匯匱匹匾匿區十千卅升午卉半卑卒卓協南博卜卞占卡卦卯印危即卵卷卸卹卻卿厄厘厚厝原厥厭厲去參又叉及友反叔取受叛叟叡叢口古句另叨叩只叫召叭叮可台叱史右司叻叼吁吃各合吉吊吋同名后吏吐向吒君吝吞吟吠否吧吩含听吳吵吶吸吹吻吼吾呀呂呃呆呈告呎呢呤周味呵呸呻呼命咀咁咄咆和咎咐咒咕咖咚咧咨咩咪咫咬咭咯咱咳咸咽哀品哄哆哇哈哉員哥哦哨哩哪哭哮哲哺哼哿唁唄唆唇唉唐唑唔唧唬售唯唱唸唾啃啄商啊問啕啞啟啡啣啤啥啦啶啼啾喀喂喃善喇喉喊喋喔喘喙喚喜喝喟喧喪喫喬單喱喻嗅嗎嗓嗔嗜嗟嗡嗣嗦嗩嗯嗲嗶嗽嘆嘈嘉嘌嘎嘔嘖嘗嘛嘧嘩嘮嘯嘲嘴嘸嘻嘿噁噎器噩噪噬噴噶噸噹嚇嚎嚏嚐嚥嚨嚮嚴嚼囂囉囊囍囑囚四回因囤困囹固圃圄圈國圍園圓圖團圜土在圩圭圮地圳圻圾址均坊坌坍坎坏坐坑坡坤坦坨坪坭坳坷坻垂垃型垓垚垛垠垢垣垮埂埃埋城埏埒埔埕域埠埤埭執埸培基埼堀堂堅堆堇堉堊堍堠堡堤堪堯堰報場堵塊塌塑塔塗塘塚塞塢塤填塵塾墀境墅墉墊墓墘墜增墟墨墩墮墳墾壁壅壇壑壓壕壘壙壞壟壢壤壩士壬壯壹壺壼壽夏夔夕外夙多夜夠夢夥大天太夫夭央夯失夷夸夾奄奇奈奉奎奏奐契奔奕套奘奚奠奢奧奪奭奮女奴奶奸她好如妃妄妊妍妒妓妖妙妝妞妣妤妥妨妮妲妳妹妻妾姆姊始姐姑姒姓委姚姜姝姣姥姦姨姪姬姮姻姿威娃娑娘娛娜娟娠娣娥娩娶娼婁婆婉婕婚婢婦婧婪婭婷婺婿媒媚媛媧媲媳媽嫁嫂嫉嫌嫖嫚嫡嫣嫦嫩嫻嬉嬋嬌嬗嬛嬤嬪嬬嬰嬴嬸嬿孀孃孌子孔孕孖字存孚孛孜孝孟孢季孤孩孫孱孵學孺孽孿宁它宅宇守安宋完宏宓宕宗官宙定宛宜客宣室宥宦宮宰害宴宵家宸容宿寂寄寅密寇富寐寒寓寞察寡寢寥實寧寨審寫寬寮寰寵寶寸寺封射將專尉尊尋對導小少尕尖尚尤尪尬就尷尸尹尺尻尼尾尿局屁居屆屈屋屍屎屏屑展屜屠屢層履屬屯山岌岐岑岔岡岩岫岬岱岳岷岸峇峋峒峙峨峪峭峰島峻峽崁崆崇崎崑崔崖崗崙崚崛崞崢崤崧崩崮崴崽嵊嵌嵎嵐嵩嵬嵯嶄嶇嶙嶲嶷嶺嶼嶽巍巒巔巖川州巡巢工左巧巨巫差巰己已巳巴巷巽巾市布帆希帑帕帖帘帚帛帝帥師席帳帶帷常帽幄幅幌幕幗幘幟幡幢幣幫干平年并幸幹幻幼幽幾庄庇床庋序底庖店庚府庠度座庫庭庵庶康庸庹庾廁廂廈廉廊廒廓廕廖廚廝廟廠廡廢廣廩廬廳延廷建廿弁弄弈弊弋式弒弓弔引弗弘弛弟弢弦弧弩弱張強弼彆彈彊彌彎彗彘彙形彤彥彧彩彪彬彭彰影彷役彼彿往征待徇很徊律後徐徑徒得徘徙從徠御徨復循徬徭微徵德徹徽心必忌忍忐忑忒忖志忘忙忠忡忤快忭忱念忸忻忽忿怎怒怕怖思怠怡急性怨怩怪怯恁恂恃恆恉恍恐恕恙恢恣恤恥恨恩恪恫恬恭息恰恿悄悅悉悌悍悔悖悚悝悟悠患您悱悲悵悶悸悼悽情惆惇惊惋惑惕惚惜惟惠惡惦惰惱想惶惹惺惻愁愆愈愉愍意愕愚愛感愧愫愷愿慄慈態慌慎慕慘慚慟慢慣慧慨慫慮慰慵慶慷慾憂憊憎憐憑憚憤憧憨憩憫憬憲憶憾懂懃懇懈應懊懋懣懦懲懶懷懸懺懼懾懿戀戇戈戊戌戍戎成我戒戕或戚戛戟戡截戮戰戲戳戴戶戾房所扁扇扈手才扎扑扒打扔托扛扣扦扭扮扯扳扶批扼找承技抃抄抉把抑抒抓投抖抗折抨披抬抱抵抹押抽拂拄拆拇拈拉拋拌拍拎拐拒拓拔拖拗拘拙拚招拜括拭拮拯拱拳拴拷拼拽拾拿持挂指按挑挖挨挪挫振挹挺挽挾捂捅捆捉捍捎捏捐捕捧捨捩捫据捱捲捶捷捺捻掀掃掄授掉掌掎掏掐排掖掘掙掛掟掠採探掣接控推掩措掰掾揀揆揉揍描提插揖揚換握揣揩揪揭揮援揹損搏搐搓搖搗搜搞搡搥搪搬搭搶摒摔摘摧摩摭摯摶摸摹摺摻摽撂撇撈撐撒撓撕撞撤撥撩撫撬播撮撰撲撻撼撾撿擁擂擄擅擇擊擋操擎擒擔擘據擠擢擦擬擭擱擲擴擺擾攀攄攏攔攘攙攜攝攣攤攪攫攬支收攸改攻放政故效敏救敔敕敖敗敘教敝敞敢散敦敬敲整敵敷數斂斃文斌斐斑斗料斛斜斟斡斤斥斧斫斬斯新斷方於施旁旃旄旅旋旌族旒旗既日旦旨早旬旭旱旺旻昀昂昃昆昇昊昌明昏易昔昕昝星映昤春昧昨昭是昱昴昵昶時晃晉晏晒晙晚晝晞晟晤晦晨普景晰晴晶晸智晾暄暇暈暉暌暐暑暖暗暝暠暢暨暫暮暱暴暹曄曆曉曖曙曜曝曠曦曩曬曰曲曳更書曹曼曾替最會月有朋服朔朕朗望朝期木未末本札朱朴朵朽杆杉李杏材村杖杜杞束杠杭杯杰東杲杳杵杷杻杼松板枇枉枋析枕林枚果枝枯架枷枸柁柄柏某柑染柔柘柚柜柝柞柢查柩柬柯柰柱柳柴柵柿栓栖栗栝校栩株栲核根格栽桀桂桃桅框案桉桌桎桐桑桓桴桶桿梁梅梆梏梓梗條梟梢梧梨梭梯械梳梵棄棉棋棍棐棒棕棗棘棚棟棠棣棧棨棪棫森棱棲棵棹棺棻棼椅椋植椎椏椒椰椴椿楊楓楔楙楚楞楠楢楣楨楫業極楷楹概榆榎榔榕榛榜榨榭榮榴榷榻榿槃槊構槍槎槐槓槙槤槭槳槻槽槿樁樂樊樑樓樗標樞樟模樣樵樸樹樺樽橄橇橈橋橘橙機橡橢橫橿檀檄檎檐檔檗檜檢檬檯檳檸檻櫃櫓櫚櫛櫟櫥櫧櫨櫬櫸櫻欄權欒欖欠次欣欲欹欺欽款歃歆歇歉歌歎歐歙歜歟歡止正此步武歧歪歲歷歸歹死殂殃殆殉殊殖殘殫殭殮殯殲段殷殺殼殿毀毅毆毋母每毒毓比毖毗毘毛毫毬毯毽氏氐民氓气氘氙氚氛氟氣氦氧氨氫氬氮氯氰水永氾汀汁求汊汎汐汕汗汙汛汜汝汞江池污汨汪汰汲汴汶決汽汾沁沂沃沅沈沉沌沐沒沓沔沖沙沚沛沫沭沮沱河沸油治沼沽沾沿況泄泅泉泊泌泓法泖泗泚泛泡波泣泥注泫泮泰泳泵洄洋洌洒洗洙洛洞津洩洪洮洱洲洵洶洸洹活洽派流浙浚浦浩浪浮浴海浸涂涅涇消涉涌涕涪涮涯液涵涼涿淀淄淅淆淇淋淌淑淒淖淘淙淚淞淡淤淦淨淪淫淮淯深淳淵混淹淺添淼清渚減渝渠渡渣渤渥渦測渭港渲渴游渺渾湃湄湊湍湖湘湛湜湟湣湧湮湯溉源準溘溜溝溟溢溥溧溪溫溯溲溴溶溺溼溽滁滂滄滅滇滋滌滎滑滓滔滕滘滬滯滲滴滷滸滾滿漁漂漆漏漓演漕漠漢漣漩漪漫漬漱漲漳漶漸漾漿潁潑潔潘潛潞潟潢潤潭潮潯潰潸潼澀澂澄澆澈澍澎澔澗澡澤澧澮澱澳澹激濁濂濃濉濕濘濛濟濠濡濤濫濬濮濰濱濺濾瀆瀉瀋瀏瀑瀕瀘瀚瀛瀝瀟瀦瀧瀨瀾灌灑灘灝灞灣火灰灶灸灼災炅炊炎炒炕炖炙炤炫炬炭炮炯炳炸為烜烤烯烴烷烹烽焉焊焙焚焜無焦焯焰然煃煇煉煌煎煒煙煜煞煤煥照煩煬煮煲煽熄熊熏熒熔熙熟熨熬熱熵熹熾燁燃燄燈燉燊燎燒燕燙營燥燦燧燬燭燮燹燼燾爆爇爍爐爛爪爬爭爰爵父爸爹爺爽爾牂牆片版牌牒牘牙牛牟牠牡牢牧物牲特牽犀犁犍犒犛犢犧犬犯狀狂狄狌狐狗狙狠狡狩狸狹狼狽猁猖猗猛猜猝猞猥猩猴猶猷猾猿獄獅獎獗獠獨獲獵獷獸獻獼獾玄率玉王玕玖玟玠玥玦玩玫玲玳玷玻珀珂珈珊珍珙珞珠珩珪班珮珽現球琅理琇琉琊琍琚琛琢琥琦琨琪琮琯琰琳琴琵琶瑁瑋瑕瑙瑚瑛瑜瑞瑟瑣瑤瑩瑪瑭瑯瑰瑾璀璃璆璇璉璋璐璘璜璞璟璦璧璨璫環璽璿瓊瓏瓚瓜瓢瓣瓦瓮瓶瓷甄甕甘甚甜生產甥甦用甩甫甬甯田由甲申男甸町甾畀畈畋界畏畔留畜畝畢畤略畦番畫異當畸畿疆疇疊疏疑疙疚疝疣疤疥疫疲疳疵疹疼疾病症痊痍痔痕痘痙痛痞痢痣痰痲痳痴痹痺痿瘀瘁瘉瘋瘍瘓瘟瘠瘡瘤瘦瘧瘩瘳瘴療癆癌癒癘癟癡癢癬癮癱癸登發白百皂的皆皇皈皋皎皓皖皚皮皰皺皿盂盃盅盆盈益盎盒盔盛盜盞盟盡監盤盥盧盪目盯盱盲直相盼盾省眈眉看眙真眠眨眩眭眯眶眷眸眺眼眾睇睛睜睞睡睢督睦睪睫睬睹睽睾睿瞄瞇瞋瞎瞑瞞瞢瞧瞪瞬瞭瞰瞳瞻瞿矗矚矛矜矢矣知矩短矮矯石矻矽砂砆砌砍研砝砟砢砦砧砭砲破砵砸硃硅硒硝硤硨硫硬确硯硼硿碇碌碎碑碓碗碘碚碟碣碧碩碭碰碳碴確碻碼碾磁磅磊磋磐磔磚磡磧磨磬磯磲磷磺礁礎礑礙礦礪礫示礽社祀祁祂祇祈祉祊祐祕祖祗祚祛祜祝神祟祠祥祧票祭祺祿禁禍禎福禕禦禧禪禮禱禳禹禺禽禾禿秀私秉秋科秒秕秘租秣秤秦秧秩秫秸移稀稃稅程稍稔稗稙稚稜稞稟稠種稱稷稻稼稽稿穀穆穌積穎穗穢穩穫穴究穹空穿突窄窆窈窒窕窖窗窘窟窠窨窩窪窮窯窺窿竄竅竇竊立竑站竟章竣童竭端競竹竺竽竿笈笏笑笘笙笛笞笠符笨第笭筅筆等筊筋筍筏筐筑筒答策筠筩筮筱筲筵筷箇箋箍箏箔箕算管箬箭箱箴箸節範篆篇築篙篚篠篡篤篦篩篷篾簇簋簑簡簧簪簷簸簽簾簿籀籃籌籍籙籠籤籬籲米籽粉粑粒粕粗粘粟粥粱粲粵粹粽精粿糊糌糕糖糙糜糞糟糠糧糯糰糴糸系糾紀約紅紆紉紊紋納紐紓純紗紘紙級紛紜素紡索紫紮累細紳紹紺絀終絃組結絕絛絜絞絡絢給絨絮統絲絳絹綁綏綑經綜綠綢綦綬維綱網綴綵綸綺綻綽綾綿緊緋緒線緝緞締緣編緩緬緯練緻縉縊縑縛縝縞縣縫縮縯縱縷總績繁繃繆織繕繖繞繡繩繪繫繭繳繹繼纂纇纈續纏纓纖纘纛纜缶缸缺缽罄罌罐罔罕罘罟罩罪置罰署罵罷罹羅羆羈羊羌美羔羚羞群羥羧羨義羯羰羲羸羹羽羿翁翅翊翌翎習翔翟翠翡翥翦翩翮翰翱老考耄者耆耋而耍耐耒耕耖耗耘耙耜耦耳耶耽耿聆聊聒聖聘聚聞聯聰聲聳聶職聽聾聿肄肅肆肇肉肋肌肖肘肚肛肜肝股肢肥肩肪肫肯肱育肺胂胃胄背胎胖胚胛胞胡胤胥胭胯胰胱胸胺能脂脅脆脈脊脖脛脣脩脫脹脾腆腊腋腌腎腐腑腓腔腕腥腦腧腫腰腱腳腴腸腹腺腿膀膂膈膊膏膚膛膜膝膠膨膩膳膺膽膾膿臀臂臃臆臉臊臘臚臞臟臣臥臧臨自臬臭至致臺臻臼臾舀舂舅與興舉舊舌舍舒舔舖舜舞舟航般舵舶舷舸船舺艇艮良艱色艷艾芊芋芍芎芒芙芝芡芥芩芫芬芭芮芯花芳芷芸芹芽芾苑苒苓苔苕苗苛苜苞苟苡苣若苦苧苯英苳苹苻苾茁茂范茄茅茆茉茌茗茛茜茨茫茯茱茲茴茵茶茸茹荀荃荅草荊荏荐荒荔荖荷荸荻荼荽莆莉莊莎莒莓莖莘莞莢莪莫莽菀菁菅菇菊菌菏菑菘菜菠菡菩華菱菲菴菸萁萃萄萇萊萌萍萎萩萬萱萸萼落葆葉著葛葡董葦葫葬葳葵葶葺蒂蒐蒔蒙蒜蒞蒡蒨蒯蒲蒴蒸蒺蒼蒿蓀蓄蓆蓉蓋蓑蓓蓬蓮蓼蔑蔓蔗蔚蔡蔣蔥蔬蔭蔻蔽蕃蕈蕉蕊蕙蕞蕨蕩蕪蕭蕾薄薇薈薊薑薙薛薜薦薨薩薪薯薰薹薺藉藍藏藐藕藜藝藤藥藨藩藪藻蘄蘅蘆蘇蘊蘋蘑蘚蘧蘭蘸蘼蘿虎虐虔處虖虛虜虞號虧虫虱虹蚊蚌蚓蚕蚜蚣蚤蚨蚩蚪蚵蚺蛄蛆蛇蛉蛋蛐蛔蛙蛛蛟蛤蛩蛭蛹蛺蛻蛾蜀蜂蜆蜈蜊蜑蜒蜓蜘蜚蜜蜡蜥蜱蜴蜷蜻蜿蝌蝕蝗蝘蝙蝠蝦蝮蝴蝶蝸螂螃螈融螞螟螢螭螯螳螺螽蟀蟄蟆蟈蟋蟑蟠蟬蟲蟹蟻蟾蠃蠅蠆蠍蠑蠓蠔蠕蠟蠡蠢蠣蠱蠲蠶蠻血行衍術街衙衛衝衡衢衣表衫衰衷衹袁袋袍袒袖袛袞袪被袱裁裂裔裕裘裙補裝裡裨裱裲裳裴裸裹製裾褂複褐褒褚褡褥褪褫褲褶褸褻襄襖襟襠襤襦襪襬襯襲西要覃覆見規覓視覘親覲覺覽觀角觔觚觝解觴觸觿言訂訃訇計訊訌討訐訓訕訖託記訛訝訟訢訣訥訪設許訴訶診註証詐詒詔評詗詛詞詠詢試詩詫詬詭詮詰話該詳詹詼誅誇誌認誓誕誘語誠誡誣誤誥誦誨說誰課誼調諂談請諍諏諒論諛諜諡諤諦諧諫諭諮諱諳諷諸諺諾謀謁謂謄謇謊謎謗謙謚講謝謠謨謫謬謳謹謾譁證譏識譙譚譜警譬譯議譴護譽讀變讓讖讙讚讞谷谿豁豆豈豉豊豌豎豐豔豚象豢豹豺貂貉貊貌貍貓貝貞負財貢貧貨販貪貫責貴貶買貸費貼貽貿賀賁賂賃賄資賈賊賑賓賚賜賞賠賡賢賣賤賦質賬賭賴賸賺購賽贇贈贊贏贓贖贛赤赦赫赭走赴赶起趁超越趕趙趟趣趨足趴趵趾跆跋跌跑跖跗跛距跟跡跣跤跨跪路跳踊踏踐踝踞踢踩踰踴踵踹蹂蹄蹇蹈蹊蹋蹔蹟蹠蹤蹦蹬蹭蹯蹲蹴蹶蹼躁身躬躲躺軀車軋軌軍軒軔軛軟軫軸軾較輅載輒輓輔輕輛輜輝輟輩輪輯輸輻輾輿轂轄轅轉轍轎轟辛辜辟辣辦辨辭辯辰辱農迂迄迅迎近返迢迤迥迦迨迪迫迭述迴迷迺追退送适逃逄逅逆逋逍透逐逑途逕逗這通逛逝逞速造逢連逮逯週進逵逶逸逼逾遁遂遇遊運遍過遏遐遑遒道達違遘遙遜遞遠遣遨適遭遮遲遴遵遶遷選遺遼遽避邀邁邂還邇邈邊邏邑邕邛邠邢那邦邪邯邰邱邳邴邵邸邽邾郁郃郅郇郊郎郛郜郝郡郢部郭郯郴郵都郾鄂鄉鄒鄔鄙鄞鄧鄭鄯鄰鄱鄴鄺酆酉酊酋酌配酐酒酗酚酣酥酩酪酬酮酯酴酵酷酸醅醇醉醋醍醐醒醚醛醜醞醣醫醬醮醯醴釀釁采釉釋里重野量釐金釗釘釜針釣釦釧釵釷鈉鈍鈔鈕鈞鈣鈴鈷鈸鈺鈾鉀鉅鉉鉑鉗鉚鉛鉞鉤鉬鉸鉻銀銃銅銓銖銘銜銥銨銫銳銷銼鋁鋅鋆鋒鋤鋪鋰鋸鋼錄錐錒錕錘錚錠錡錢錦錨錫錮錯錳錶鍊鍋鍍鍛鍥鍬鍰鍱鍵鍶鍼鍾鎂鎊鎌鎖鎗鎚鎡鎧鎩鎬鎮鎳鏈鏖鏗鏘鏞鏟鏡鏢鏤鏽鐘鐙鐮鐵鐸鑄鑊鑑鑒鑣鑫鑰鑱鑲鑷鑼鑽鑾鑿長門閂閃閉開閏閑閒間閔閘閡閣閤閥閨閩閫閭閱閹閻閼閾闆闇闈闊闌闓闔闕闖關闞闡闢阜阡阪阮阱防阻阽阿陀陂附陋陌降限陛陝陞陟陡院陣除陪陰陲陳陵陶陷陸陽隄隅隆隈隊隋隍階隔隕隗隘隙際障隧隨險隱隴隸隹隻隼雀雁雄雅集雇雉雋雌雍雎雒雕雖雘雙雛雜雞離難雨雪雯雰雲零雷雹電需霄霆震霉霍霎霏霑霓霖霙霜霞霧霨霰露霸霹霽霾靂靄靈青靖靚靛靜非靠靡面靨革靳靴靶靼鞅鞋鞍鞏鞘鞠鞣鞭韁韃韋韌韓韜韞韭音韶韻響頁頂頃項順須頊頌頏預頑頒頓頗領頜頡頤頦頭頰頷頸頹頻顆題額顎顏顒顓願顙顛類顥顧顫顯顱風颯颱颶飄飆飛食飢飩飪飭飯飲飴飼飽飾餅餉養餌餐餒餓餘餚餛餞餡館餬餵餽餾饅饈饋饌饑饒饕饗首馗香馥馨馬馭馮馳馴駁駐駒駕駙駛駝駭駱駿騁騎騏騑騙騧騫騰騷騾驁驃驄驅驊驍驕驗驚驛驟驢驤驩驪骨骰骷骸骼髀髏髒髓體髖高髦髮髯髻鬃鬆鬍鬘鬚鬟鬣鬥鬧鬩鬯鬱鬲鬻鬼魁魂魄魅魋魏魔魚魨魯魴魷鮆鮑鮚鮠鮨鮪鮫鮭鮮鮸鯁鯉鯊鯔鯖鯛鯡鯧鯨鯰鯽鰈鰍鰓鰨鰭鰲鰻鱄鱈鱔鱗鱘鱠鱧鱨鱷鱸鱺鳥鳩鳳鳴鳶鴉鴒鴛鴞鴦鴨鴻鴿鵝鵠鵡鵪鵬鵰鵲鶉鶩鶯鶴鶿鷗鷳鷸鷹鷺鸕鸚鸛鸞鹵鹹鹼鹽鹿麂麇麋麒麓麗麝麟麥麩麴麵麻麼麾黃黎黏黑黔默黛黜點黠黧黨黯黴黷黼鼎鼐鼓鼠鼬鼻齊齋齒齡齣齦齧龍龐龔龕龜龢"
#     elif lang in ['hi', 'mr']:
#         opt.saved_model = r"D:\OCR\saved_models\best_accuracy_hi.pth"
#         opt.character = '0123456789!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.ँंःअअंअःआइईउऊऋएऐऑओऔकखगघङचछजझञटठडढणतथदधनऩपफबभमयरऱलळवशषसह़ािीुूृॅेैॉोौ्ॐ॒क़ख़ग़ज़ड़ढ़फ़ॠ।०१२३४५६७८९॰'
#     elif lang == 'ha_hi':
#         opt.saved_model = r"D:\OCR\saved_models\hindi_handwritten.pth"
#         opt.character = '0123456789!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.ँंःअअंअःआइईउऊऋएऐऑओऔकखगघङचछजझञटठडढणतथदधनऩपफबभमयरऱलळवशषसह़ािीुूृॅेैॉोौ्ॐ॒क़ख़ग़ज़ड़ढ़फ़ॠ।०१२३४५६७८९॰'
#  # same with ASTER setting (use 94 char).




    


