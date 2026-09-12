package main

import (
	"bufio"
	"fmt"
	"math"
	"math/rand"
	"os"
	"sort"
	"strconv"
	"strings"
)

var cnDigits = []string{"零", "一", "二", "三", "四", "五", "六", "七", "八", "九"}

func cnNum(n int) string {
	switch {
	case n < 10:
		return cnDigits[n]
	case n < 20:
		if n%10 == 0 {
			return "十"
		}
		return "十" + cnDigits[n%10]
	case n < 100:
		t, o := n/10, n%10
		if o == 0 {
			return cnDigits[t] + "十"
		}
		return cnDigits[t] + "十" + cnDigits[o]
	}
	return strconv.Itoa(n)
}

// ---------------------------------------------------------------- 语料

var (
	person = []string{"我", "你", "他", "她", "我们", "你们", "大家", "小王", "小李", "老师", "朋友"}
	timew  = []string{"今天", "明天", "昨天", "早上", "中午", "晚上", "周末", "下周", "刚才", "现在"}
	place  = []string{"公司", "学校", "家里", "公园", "超市", "图书馆", "食堂", "车站", "医院", "健身房"}
	obj    = []string{"电脑", "手机", "书", "咖啡", "作业", "报告", "方案", "行李", "雨伞", "自行车", "相机", "耳机", "键盘", "外套", "午餐", "门票"}
	verb   = []string{"准备", "整理", "看完", "写完", "买好", "带上", "检查", "收拾", "研究", "练习"}
	adj    = []string{"开心", "顺利", "糟糕", "有趣", "累", "紧张", "轻松", "满意", "麻烦", "重要"}
	food   = []string{"面", "米饭", "火锅", "烧烤", "沙拉", "饺子", "寿司", "粥", "汉堡", "炒饭", "麻辣烫", "烤鱼", "汤面", "三明治", "包子", "盖饭"}
	emo    = []string{"高兴", "难过", "焦虑", "平静", "兴奋", "疲惫", "放松", "烦躁", "踏实", "孤单"}
	topic  = []string{"学习", "工作", "运动", "旅行", "读书", "编程", "做饭", "摄影", "音乐", "健身", "理财", "睡眠", "时间管理", "交朋友", "学英语", "写代码"}
)

type pair struct{ q, a string }

func pick(r *rand.Rand, s []string) string { return s[r.Intn(len(s))] }

func genSlots(r *rand.Rand, n int) []pair {
	out := make([]pair, 0, n)
	for i := 0; i < n; i++ {
		p, t, pl, o := pick(r, person), pick(r, timew), pick(r, place), pick(r, obj)
		v, a, f, e := pick(r, verb), pick(r, adj), pick(r, food), pick(r, emo)
		var q, ans string
		switch r.Intn(10) {
		case 0:
			q = t + "你打算做什么"
			ans = t + "我" + pick(r, verb) + "去" + pl + "，顺便把" + o + pick(r, verb) + "好。"
		case 1:
			q = "为什么" + p + t + "看起来有点" + e
			ans = "因为" + t + "的事情不太" + a + "，不过现在好多了。"
		case 2:
			q = p + "在" + pl + "做什么"
			ans = p + "在" + pl + pick(r, verb) + o + "，看起来挺" + a + "的。"
		case 3:
			q = t + "吃什么比较好"
			ans = "要不试试" + f + "？" + t + "吃这个挺合适的。"
		case 4:
			q = "怎么才能把" + o + pick(r, verb) + "好"
			ans = "先列个清单，再一件一件来，别一次做太多。"
		case 5:
			q = p + "觉得" + pick(r, topic) + "重要吗"
			ans = "挺重要的，坚持下来会有明显变化。"
		case 6:
			q = "最近" + pick(r, topic) + "怎么样"
			ans = "还行，每天花一点时间，慢慢就有感觉了。"
		case 7:
			q = t + "的天气适合去" + pl + "吗"
			ans = "适合，不过记得看看天气，带把伞更稳妥。"
		case 8:
			q = p + t + "心情怎么样"
			ans = t + "心情还算" + a + "，就是有点" + e + "。"
		default:
			q = o + "放在哪里了"
			ans = "应该在" + pl + "那边，你找找看。"
		}
		out = append(out, pair{q, ans})
		_ = v
	}
	return out
}

var rounds = [][]string{
	{"你好", "你好呀，今天过得怎么样？", "还不错，你呢？", "我也挺好的，谢谢你关心。"},
	{"在忙吗", "刚忙完一阵，现在有空。", "那正好，我想问你个事。", "你说吧，我听着。"},
	{"我最近在学编程", "挺好的呀，学的什么语言？", "在学python，感觉还行。", "python入门很合适，多写代码就有感觉了。"},
	{"今天好累", "怎么了，是工作太忙吗？", "是啊，事情堆了一大堆。", "那先歇一会儿，事情一件一件做就好。"},
	{"周末有什么打算", "想去公园走走，顺便拍点照片。", "听起来不错，天气好的话很舒服。", "是啊，正好放松一下。"},
	{"你会做饭吗", "会一点简单的，比如煮面和炒饭。", "那挺厉害的，我经常点外卖。", "外卖偶尔吃可以，还是自己做的干净。"},
	{"最近睡不好", "是压力大还是作息乱？", "可能是想的事情太多了。", "睡前别看手机，早点躺下会好一些。"},
	{"我想换个工作", "是现在这份不太满意吗？", "嗯，感觉没什么成长。", "那可以边工作边找，别急着裸辞。"},
	{"你平时看书吗", "看，最近在看一本讲心理学的。", "有意思吗？我也想找点书看。", "挺有意思的，能解释不少日常现象。"},
	{"减肥好难", "别太急，慢慢来才不容易反弹。", "可是效果太慢了。", "每周稳定掉一点就已经很不错了。"},
}

func genDialogue(r *rand.Rand, n int) []pair {
	out := make([]pair, 0, n)
	for i := 0; i < n; i++ {
		d := rounds[r.Intn(len(rounds))]
		for j := 0; j+1 < len(d); j += 2 {
			out = append(out, pair{d[j], d[j+1]})
		}
	}
	return out
}

var know = [][2]string{
	{"什么是大模型", "大模型是通过学习海量文本，学会预测下一个词的程序。"},
	{"什么是token", "token是文本切分后的最小单元，会被映射成数字ID。"},
	{"什么是分词器", "分词器负责把句子切成token，再查表变成数字。"},
	{"什么是神经网络", "神经网络是一堆带参数的函数叠起来，靠梯度下降调参数。"},
	{"什么是注意力机制", "注意力让每个token去看其他token，决定该关注谁。"},
	{"什么是transformer", "transformer是堆叠注意力和前馈网络的结构。"},
	{"什么是过拟合", "过拟合是模型把训练数据背下来了，换新数据就不行。"},
	{"什么是梯度下降", "梯度下降是沿着损失下降的方向一点点调参数。"},
	{"什么是embedding", "embedding把离散的ID映射成稠密向量，让语义可以计算。"},
	{"什么是学习率", "学习率决定每次参数更新的步子迈多大。"},
	{"什么是损失函数", "损失函数衡量预测和真实答案差多远。"},
	{"什么是反向传播", "反向传播用链式法则把误差一层层传回去算梯度。"},
	{"什么是参数", "参数就是模型里那些可以被训练调整的数字。"},
	{"什么是推理", "推理是用训练好的参数对新的输入算输出。"},
	{"怎么学编程", "从一个小项目开始，边做边学最快。"},
	{"怎么学英语", "每天坚持一点，多听多读，比突击有效。"},
	{"怎么提高效率", "先做最重要的一件事，剩下的按顺序来。"},
	{"怎么保持专注", "把手机放远一点，一次只做一件事。"},
	{"怎么缓解焦虑", "先把担心的事写下来，再一件件拆开看。"},
	{"怎么养成习惯", "从特别小的一步开始，小到不可能失败。"},
	{"中国的首都是哪里", "中国的首都是北京。"},
	{"一年有多少天", "一年有三百六十五天，闰年三百六十六天。"},
	{"一天有多少小时", "一天有二十四小时。"},
	{"一小时多少分钟", "一小时有六十分钟。"},
	{"水多少度结冰", "水在零度结冰，一百度沸腾。"},
	{"地球是什么形状", "地球是一个两极稍扁的球体。"},
	{"彩虹有几种颜色", "彩虹有七种颜色，红橙黄绿青蓝紫。"},
	{"世界上最高的山", "世界上最高的山是珠穆朗玛峰。"},
	{"光速是多少", "光在真空中的速度大约是每秒三十万公里。"},
	{"人为什么要睡觉", "睡觉能让大脑整理记忆，身体也能恢复。"},
	{"为什么要喝水", "水参与身体几乎所有代谢，缺了会很难受。"},
	{"为什么要运动", "运动能让心肺更强，情绪也更稳定。"},
	{"为什么要读书", "读书能把别人的经验变成自己的。"},
	{"为什么会下雨", "水汽遇冷凝结成云，云里水滴变大就落下来。"},
	{"为什么天是蓝的", "空气把蓝光散射得比其他颜色多，所以看起来是蓝的。"},
	{"为什么会有四季", "因为地球自转轴是斜的，太阳照射角度在变。"},
}

func genKnow(r *rand.Rand, n int) []pair {
	out := make([]pair, 0, n)
	for i := 0; i < n; i++ {
		k := know[r.Intn(len(know))]
		out = append(out, pair{k[0], k[1]})
	}
	return out
}

func genMath(r *rand.Rand, n int) []pair {
	out := make([]pair, 0, n)
	for i := 0; i < n; i++ {
		var a, b, res int
		var op, q string
		switch r.Intn(3) {
		case 0:
			a, b = r.Intn(99)+1, r.Intn(99)+1
			res, op = a+b, "加"
			q = cnNum(a) + "加" + cnNum(b) + "等于几"
		case 1:
			a, b = r.Intn(99)+1, r.Intn(99)+1
			if a < b {
				a, b = b, a
			}
			res, op = a-b, "减"
			q = cnNum(a) + "减" + cnNum(b) + "等于几"
		default:
			a, b = r.Intn(19)+1, r.Intn(19)+1
			res, op = a*b, "乘"
			q = cnNum(a) + "乘以" + cnNum(b) + "等于几"
		}
		_ = op
		out = append(out, pair{q, strings.TrimSuffix(q, "等于几") + "等于" + cnNum(res) + "。"})
	}
	return out
}

func genCompare(r *rand.Rand, n int) []pair {
	out := make([]pair, 0, n)
	for i := 0; i < n; i++ {
		a, b := r.Intn(99)+1, r.Intn(99)+1
		var ans string
		switch {
		case a > b:
			ans = cnNum(a) + "比" + cnNum(b) + "大。"
		case a < b:
			ans = cnNum(a) + "比" + cnNum(b) + "小。"
		default:
			ans = cnNum(a) + "和" + cnNum(b) + "一样大。"
		}
		out = append(out, pair{cnNum(a) + "和" + cnNum(b) + "哪个大", ans})
	}
	return out
}

func genEmotion(r *rand.Rand, n int) []pair {
	qs := []string{"%s感觉有点%s", "我%s", "%s说%s", "最近总是%s", "为什么我会这么%s", "怎么才能不那么%s"}
	as := []string{
		"这很正常，%s的时候先接纳自己的状态。",
		"别急着赶走%s，它只是提醒你该歇一歇了。",
		"试着深呼吸几次，%s会慢慢淡下去。",
		"可以先做点小事，让注意力从%s上移开。",
		"说出来就好多了，我在听。",
		"每个人都会有%s的时候，你不孤单。",
	}
	out := make([]pair, 0, n)
	for i := 0; i < n; i++ {
		e, t, p := pick(r, emo), pick(r, timew), pick(r, person)
		tpl := qs[r.Intn(len(qs))]
		var q string
		switch strings.Count(tpl, "%s") {
		case 2:
			q = fmt.Sprintf(tpl, t, e)
		default:
			if strings.HasPrefix(tpl, "我") || strings.HasPrefix(tpl, "最近") || strings.HasPrefix(tpl, "为什么") || strings.HasPrefix(tpl, "怎么") {
				q = fmt.Sprintf(tpl, e)
			} else {
				q = fmt.Sprintf(tpl, p, e)
			}
		}
		at := as[r.Intn(len(as))]
		var ans string
		if strings.Contains(at, "%s") {
			ans = fmt.Sprintf(at, e)
		} else {
			ans = at
		}
		out = append(out, pair{q, ans})
	}
	return out
}

func buildCorpus(r *rand.Rand, path string) int {
	all := [](pair){}
	all = append(all, genSlots(r, 60000)...)
	all = append(all, genDialogue(r, 4000)...)
	all = append(all, genKnow(r, 20000)...)
	all = append(all, genMath(r, 20000)...)
	all = append(all, genCompare(r, 8000)...)
	all = append(all, genEmotion(r, 30000)...)
	r.Shuffle(len(all), func(i, j int) { all[i], all[j] = all[j], all[i] })

	f, err := os.Create(path)
	if err != nil {
		panic(err)
	}
	w := bufio.NewWriterSize(f, 1<<20)
	for _, p := range all {
		w.WriteString("问：")
		w.WriteString(p.q)
		w.WriteString("\n答：")
		w.WriteString(p.a)
		w.WriteString("\n")
	}
	w.Flush()
	f.Close()
	return len(all)
}

// ---------------------------------------------------------------- 统计

type Stats struct {
	Chars []rune
	Index map[rune]int
	Trans [][]float64 // 转移计数
	QA    []pair
	Inv   map[rune][]int32
}

func loadText(path string) string {
	b, err := os.ReadFile(path)
	if err != nil {
		panic(err)
	}
	return string(b)
}

func buildStats(text string) *Stats {
	runes := []rune(text)
	set := map[rune]bool{}
	for _, ch := range runes {
		set[ch] = true
	}
	chars := make([]rune, 0, len(set))
	for ch := range set {
		chars = append(chars, ch)
	}
	sort.Slice(chars, func(i, j int) bool { return chars[i] < chars[j] })

	index := make(map[rune]int, len(chars))
	for i, ch := range chars {
		index[ch] = i
	}

	n := len(chars)
	trans := make([][]float64, n)
	for i := range trans {
		trans[i] = make([]float64, n)
	}
	for i := 0; i+1 < len(runes); i++ {
		a, ok1 := index[runes[i]]
		b, ok2 := index[runes[i+1]]
		if ok1 && ok2 {
			trans[a][b]++
		}
	}

	// 问答与倒排索引
	lines := strings.Split(text, "\n")
	qa := make([]pair, 0, len(lines)/2)
	for i := 0; i+1 < len(lines); i += 2 {
		q, a := lines[i], lines[i+1]
		if strings.HasPrefix(q, "问：") && strings.HasPrefix(a, "答：") {
			qa = append(qa, pair{string([]rune(q)[1:]), string([]rune(a)[1:])})
		}
	}
	inv := make(map[rune][]int32, n)
	for qi, p := range qa {
		seen := map[rune]bool{}
		for _, ch := range p.q {
			if !seen[ch] {
				seen[ch] = true
				inv[ch] = append(inv[ch], int32(qi))
			}
		}
	}
	fmt.Printf("统计完成: %d 字, %d 问答对\n", len(chars), len(qa))
	return &Stats{Chars: chars, Index: index, Trans: trans, QA: qa, Inv: inv}
}

// ---------------------------------------------------------------- 检索

type hit struct {
	score float64
	qi    int
	q, a  string
}

func retrieve(st *Stats, query string, topk int) []hit {
	qr := []rune(strings.TrimSpace(query))
	if len(qr) == 0 {
		return nil
	}
	cnt := map[int32]int{}
	for _, ch := range qr {
		for _, qi := range st.Inv[ch] {
			cnt[qi]++
		}
	}
	if len(cnt) == 0 {
		return nil
	}
	maxc := 0
	for _, c := range cnt {
		if c > maxc {
			maxc = c
		}
	}
	cand := make([]int, 0, len(cnt))
	for qi, c := range cnt {
		if c >= maxc-1 {
			cand = append(cand, int(qi))
			if len(cand) >= 4000 {
				break
			}
		}
	}

	uq := map[rune]bool{}
	for _, ch := range qr {
		uq[ch] = true
	}
	denom := float64(max(1, len(uq)))

	hs := make([]hit, 0, len(cand))
	for _, qi := range cand {
		p := st.QA[qi]
		score := float64(cnt[int32(qi)]) / denom
		// 连续子串加分
		qrS := string(qr)
		for L := min(len(qrS), 6); L > 1; L-- {
			found := false
			for i := 0; i+L <= len(qrS); i++ {
				if strings.Contains(p.q, qrS[i:i+L]) {
					score += 0.6 * float64(L)
					found = true
					break
				}
			}
			if found {
				break
			}
		}
		// 长度差扣分
		if d := len([]rune(p.q)) - len(qr); d > 0 {
			score -= 0.02 * float64(d)
		}
		hs = append(hs, hit{score, qi, p.q, p.a})
	}
	sort.Slice(hs, func(i, j int) bool { return hs[i].score > hs[j].score })
	if len(hs) > topk {
		hs = hs[:topk]
	}
	return hs
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// ---------------------------------------------------------------- 生成

// nextToken 按相邻概率挑下一个字
func nextToken(st *Stats, cur rune, r *rand.Rand, ban map[rune]bool) (rune, bool) {
	ci, ok := st.Index[cur]
	if !ok {
		return 0, false
	}
	row := st.Trans[ci]
	total := 0.0
	for j, v := range row {
		if ban != nil && ban[st.Chars[j]] {
			continue
		}
		total += v
	}
	if total <= 0 {
		return 0, false
	}
	x := r.Float64() * total
	acc := 0.0
	for j, v := range row {
		if ban != nil && ban[st.Chars[j]] {
			continue
		}
		acc += v
		if acc >= x {
			return st.Chars[j], true
		}
	}
	return 0, false
}

// Generate 从 seed 出发，按共现概率把周围的字「框」进来
func Generate(st *Stats, seed string, maxLen int, r *rand.Rand) string {
	sr := []rune(seed)
	if len(sr) == 0 {
		return ""
	}
	cur := sr[len(sr)-1]
	var sb strings.Builder
	for i := 0; i < maxLen; i++ {
		nx, ok := nextToken(st, cur, r, nil)
		if !ok {
			break
		}
		sb.WriteRune(nx)
		cur = nx
		if strings.ContainsRune("。！？\n", cur) {
			break
		}
	}
	return sb.String()
}

// Answer 先检索框定上下文，再决定直接引用还是按概率续写
func Answer(st *Stats, query string, r *rand.Rand, verbose bool) string {
	hs := retrieve(st, query, 8)
	if len(hs) == 0 || hs[0].score < 0.5 {
		return "这个我还不太会说，教教我吧。"
	}
	if verbose {
		fmt.Println("--- 框住的语料上下文 ---")
		for i, h := range hs[:min(5, len(hs))] {
			fmt.Printf("  [%d] %.2f  问：%s  →  答：%s\n", i, h.score, h.q, h.a)
		}
	}
	if hs[0].score >= 1.2 || len([]rune(query)) >= 6 {
		return hs[0].a
	}
	pool := []string{}
	for _, h := range hs {
		if h.score >= hs[0].score*0.75 {
			pool = append(pool, h.a)
		}
	}
	if len(pool) == 0 {
		return hs[0].a
	}
	return pool[r.Intn(len(pool))]
}

// ---------------------------------------------------------------- 主流程

func main() {
	rand.Seed(20260912)
	r := rand.New(rand.NewSource(20260912))

	dir := "."
	if len(os.Args) > 1 && os.Args[1] == "--dir" {
		dir = os.Args[2]
		os.Args = append(os.Args[:1], os.Args[3:]...)
	}
	corpusPath := dir + "/corpus_big.txt"

	if _, err := os.Stat(corpusPath); err != nil {
		fmt.Println("生成语料库……")
		n := buildCorpus(r, corpusPath)
		fmt.Printf("生成 %d 条问答\n", n)
	}

	text := loadText(corpusPath)
	fmt.Printf("语料 %d 字符 (%.2f MB)\n", len([]rune(text)), float64(len(text))/1024/1024)

	st := buildStats(text)

	if len(os.Args) > 1 {
		q := strings.Join(os.Args[1:], " ")
		fmt.Printf("> %s\n", q)
		fmt.Println(Answer(st, q, r, true))
		fmt.Printf("(按共现概率续写: %s)\n", Generate(st, q, 30, r))
		return
	}

	fmt.Println("输入问题，Ctrl-C 退出。")
	sc := bufio.NewScanner(os.Stdin)
	for {
		fmt.Print("> ")
		if !sc.Scan() {
			break
		}
		q := strings.TrimSpace(sc.Text())
		if q == "" {
			continue
		}
		fmt.Println(Answer(st, q, r, true))
	}
	_ = math.Abs
}
