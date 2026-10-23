import os
from PIL import Image, ImageDraw, ImageFont

WIDTH = 1700
HEIGHT = 1200
BG_COLOR = (255, 255, 255)

img = Image.new("RGB", (WIDTH, HEIGHT), BG_COLOR)
draw = ImageDraw.Draw(img)

# Styling
CONTAINER_BORDER = (24, 76, 120)       # Strong Dark Navy
CONTAINER_BG = (255, 255, 255)          # Clean White
BOX_FILL = (195, 222, 248)              # Soft Crisp Sky Blue
BOX_BORDER = (24, 76, 120)              # Deep Navy
TEXT_COLOR = (10, 25, 45)               # Dark Charcoal / Black
TITLE_COLOR = (16, 68, 120)             # Heading Navy
ARROW_COLOR = (30, 45, 65)              # Crisp Slate Navy

# Fonts
font_paths = [
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/Library/Fonts/Arial.ttf"
]
font_title = None
font_box = None

for fp in font_paths:
    if os.path.exists(fp):
        try:
            font_title = ImageFont.truetype(fp, 26)
            font_box = ImageFont.truetype(fp, 17)
            break
        except Exception:
            pass

if font_title is None:
    font_title = ImageFont.load_default()
    font_box = ImageFont.load_default()

def draw_rounded_box(d, xy, fill=BOX_FILL, outline=BOX_BORDER, width=2, radius=12):
    x0, y0, x1, y1 = xy
    d.rounded_rectangle([x0, y0, x1, y1], radius=radius, fill=fill, outline=outline, width=width)

def draw_text_centered(d, text, xy, font, fill=TEXT_COLOR):
    lines = text.split('\n')
    line_heights = []
    line_widths = []
    for l in lines:
        bbox = font.getbbox(l)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        line_widths.append(w)
        line_heights.append(h + 5)
    total_h = sum(line_heights) - 5
    
    cx = (xy[0] + xy[2]) / 2
    cy = (xy[1] + xy[3]) / 2
    cur_y = cy - total_h / 2
    
    for i, l in enumerate(lines):
        w = line_widths[i]
        d.text((cx - w / 2, cur_y), l, font=font, fill=fill)
        cur_y += line_heights[i]

def draw_arrow(d, p_from, p_to, color=ARROW_COLOR, width=2, arrow_size=8):
    x0, y0 = p_from
    x1, y1 = p_to
    d.line([p_from, p_to], fill=color, width=width)
    
    import math
    angle = math.atan2(y1 - y0, x1 - x0)
    ax1 = x1 - arrow_size * math.cos(angle - math.pi / 6)
    ay1 = y1 - arrow_size * math.sin(angle - math.pi / 6)
    ax2 = x1 - arrow_size * math.cos(angle + math.pi / 6)
    ay2 = y1 - arrow_size * math.sin(angle + math.pi / 6)
    d.polygon([(x1, y1), (ax1, ay1), (ax2, ay2)], fill=color)

# Exact Width: 1580px for all containers (from X = 60 to X = 1640)
LEFT_X = 60
RIGHT_X = 1640

# =========================================================================
# 1. CONTAINER 1: DATA
# Bounds: (60, 30) to (1640, 390) (Height: 360px)
# =========================================================================
c1_box = (LEFT_X, 30, RIGHT_X, 390)
draw_rounded_box(draw, c1_box, CONTAINER_BG, CONTAINER_BORDER, width=2, radius=22)
draw.text((WIDTH / 2 - 32, 45), "DATA", font=font_title, fill=TITLE_COLOR)

# Nodes in Container 1
b_load = (90, 175, 240, 245)
draw_rounded_box(draw, b_load)
draw_text_centered(draw, "Dataset\nLoading", b_load, font_box)

b_eda = (290, 175, 470, 245)
draw_rounded_box(draw, b_eda)
draw_text_centered(draw, "Exploratory\nData Analysis", b_eda, font_box)

draw_arrow(draw, (240, 210), (290, 210))

# Fork bus at X = 515
draw.line([(470, 210), (515, 210)], fill=ARROW_COLOR, width=2)
draw.line([(515, 115), (515, 305)], fill=ARROW_COLOR, width=2)

# 3 Branches
b_pre = (555, 80, 755, 150)
draw_rounded_box(draw, b_pre)
draw_text_centered(draw, "Text\nPreprocessing", b_pre, font_box)

b_tfidf = (805, 80, 1055, 150)
draw_rounded_box(draw, b_tfidf)
draw_text_centered(draw, "TF-IDF Feature\nExtraction", b_tfidf, font_box)

draw_arrow(draw, (515, 115), (555, 115))
draw_arrow(draw, (755, 115), (805, 115))

b_ling = (665, 175, 965, 245)
draw_rounded_box(draw, b_ling)
draw_text_centered(draw, "Linguistic Feature\nExtraction", b_ling, font_box)
draw_arrow(draw, (515, 210), (665, 210))

b_vader = (665, 270, 965, 340)
draw_rounded_box(draw, b_vader)
draw_text_centered(draw, "VADER Sentiment\nFeature Extraction", b_vader, font_box)
draw_arrow(draw, (515, 305), (665, 305))

# Join bus at X = 1115
draw.line([(1055, 115), (1115, 115)], fill=ARROW_COLOR, width=2)
draw.line([(965, 210), (1115, 210)], fill=ARROW_COLOR, width=2)
draw.line([(965, 305), (1115, 305)], fill=ARROW_COLOR, width=2)
draw.line([(1115, 115), (1115, 305)], fill=ARROW_COLOR, width=2)

b_prep = (1165, 175, 1375, 245)
draw_rounded_box(draw, b_prep)
draw_text_centered(draw, "Feature\nPreparation", b_prep, font_box)
draw_arrow(draw, (1115, 210), (1165, 210))

# =========================================================================
# INTER-CONTAINER ROUTE 1 (DATA -> MODEL BUILDING)
# Travels strictly through the open gap (Y = 415)
# =========================================================================
# From bottom of Feature Preparation (1270, 245) -> (1270, 365) -> (1570, 365) -> (1570, 415) -> (185, 415) -> (185, 530)
draw.line([(1270, 245), (1270, 365)], fill=ARROW_COLOR, width=2)
draw.line([(1270, 365), (1570, 365)], fill=ARROW_COLOR, width=2)
draw.line([(1570, 365), (1570, 415)], fill=ARROW_COLOR, width=2)
draw.line([(1570, 415), (185, 415)], fill=ARROW_COLOR, width=2)
draw_arrow(draw, (185, 415), (185, 530))

# =========================================================================
# 2. CONTAINER 2: MODEL BUILDING
# Bounds: (60, 440) to (1640, 830) (Height: 390px, Gap: 50px)
# =========================================================================
c2_box = (LEFT_X, 440, RIGHT_X, 830)
draw_rounded_box(draw, c2_box, CONTAINER_BG, CONTAINER_BORDER, width=2, radius=22)
draw.text((WIDTH / 2 - 105, 455), "MODEL BUILDING", font=font_title, fill=TITLE_COLOR)

b_split = (85, 530, 285, 605)
draw_rounded_box(draw, b_split)
draw_text_centered(draw, "80:20 Stratified\nTrain-Test Split", b_split, font_box)

b_comb = (335, 530, 515, 605)
draw_rounded_box(draw, b_comb)
draw_text_centered(draw, "Feature\nCombination", b_comb, font_box)
draw_arrow(draw, (285, 567), (335, 567))

# Distribution bus at X = 560
draw.line([(515, 567), (560, 567)], fill=ARROW_COLOR, width=2)
draw.line([(560, 512), (560, 712)], fill=ARROW_COLOR, width=2)

# 3 Candidate Models with GENEROUS 25px clear spacing between them!
b_lr = (600, 485, 800, 540)
draw_rounded_box(draw, b_lr)
draw_text_centered(draw, "Logistic\nRegression", b_lr, font_box)

b_svm = (600, 565, 800, 620)
draw_rounded_box(draw, b_svm)
draw_text_centered(draw, "Linear SVM", b_svm, font_box)

b_rf = (600, 645, 800, 700)
draw_rounded_box(draw, b_rf)
draw_text_centered(draw, "Random\nForest", b_rf, font_box)

draw_arrow(draw, (560, 512), (600, 512))
draw_arrow(draw, (560, 592), (600, 592))
draw_arrow(draw, (560, 672), (600, 672))

# Collection bus at X = 845
draw.line([(800, 512), (845, 512)], fill=ARROW_COLOR, width=2)
draw.line([(800, 592), (845, 592)], fill=ARROW_COLOR, width=2)
draw.line([(800, 672), (845, 672)], fill=ARROW_COLOR, width=2)
draw.line([(845, 512), (845, 672)], fill=ARROW_COLOR, width=2)

b_eval = (890, 555, 1070, 630)
draw_rounded_box(draw, b_eval)
draw_text_centered(draw, "Model\nEvaluation", b_eval, font_box)
draw_arrow(draw, (845, 592), (890, 592))

b_best = (1120, 555, 1330, 630)
draw_rounded_box(draw, b_best)
draw_text_centered(draw, "Best Model\nSelection", b_best, font_box)
draw_arrow(draw, (1070, 592), (1120, 592))

# =========================================================================
# INTER-CONTAINER ROUTE 2 (MODEL BUILDING -> DEPLOYMENT)
# Travels strictly through the open gap (Y = 855)
# =========================================================================
# From bottom of Best Model Selection (1225, 630) -> (1225, 805) -> (1570, 805) -> (1570, 855) -> (185, 855) -> (185, 965)
draw.line([(1225, 630), (1225, 805)], fill=ARROW_COLOR, width=2)
draw.line([(1225, 805), (1570, 805)], fill=ARROW_COLOR, width=2)
draw.line([(1570, 805), (1570, 855)], fill=ARROW_COLOR, width=2)
draw.line([(1570, 855), (185, 855)], fill=ARROW_COLOR, width=2)
draw_arrow(draw, (185, 855), (185, 965))

# =========================================================================
# 3. CONTAINER 3: DEPLOYMENT
# Bounds: (60, 880) to (1640, 1140) (Height: 260px, Gap: 50px)
# =========================================================================
c3_box = (LEFT_X, 880, RIGHT_X, 1140)
draw_rounded_box(draw, c3_box, CONTAINER_BG, CONTAINER_BORDER, width=2, radius=22)
draw.text((WIDTH / 2 - 85, 895), "DEPLOYMENT", font=font_title, fill=TITLE_COLOR)

# 5 Balanced Sequential Nodes
b_app = (85, 965, 305, 1045)
draw_rounded_box(draw, b_app)
draw_text_centered(draw, "Streamlit Web\nApplication", b_app, font_box)

b_user = (370, 965, 570, 1045)
draw_rounded_box(draw, b_user)
draw_text_centered(draw, "User\nInput", b_user, font_box)
draw_arrow(draw, (305, 1005), (370, 1005))

b_pipe = (635, 965, 855, 1045)
draw_rounded_box(draw, b_pipe)
draw_text_centered(draw, "Prediction\nPipeline", b_pipe, font_box)
draw_arrow(draw, (570, 1005), (635, 1005))

b_clf = (920, 965, 1120, 1045)
draw_rounded_box(draw, b_clf)
draw_text_centered(draw, "Trained\nClassifier", b_clf, font_box)
draw_arrow(draw, (855, 1005), (920, 1005))

b_pred = (1185, 965, 1385, 1045)
draw_rounded_box(draw, b_pred)
draw_text_centered(draw, "Prediction\nOutput", b_pred, font_box)
draw_arrow(draw, (1120, 1005), (1185, 1005))

# Save
out_png = "figures/project_pipeline.png"
img.save(out_png, dpi=(300, 300))
img.save("figures/project_pipeline_corrected.png", dpi=(300, 300))
print("Successfully rendered figures/project_pipeline.png!")
