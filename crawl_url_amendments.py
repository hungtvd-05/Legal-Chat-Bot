from DrissionPage import ChromiumPage, ChromiumOptions
import time
import random
import json
import os
import re
import ddddocr
import hashlib
import math
from bs4 import BeautifulSoup, Comment
from markdownify import markdownify as md
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

MY_USERNAME= os.getenv("MY_USERNAME")
MY_PASSWORD= os.getenv("MY_PASSWORD")

BASE_URL = "https://thuvienphapluat.vn"
OUTPUT_DIR = "data_json_next"

os.makedirs(OUTPUT_DIR, exist_ok=True)
ocr = ddddocr.DdddOcr(show_ad=False)


def sanitize_filename(name):
    hash_object = hashlib.md5(name.encode('utf-8'))
    return hash_object.hexdigest()


def init_browser():
    co = ChromiumOptions()
    # co.set_browser_path('/usr/bin/brave')

    # co.set_argument('--disable-blink-features=AutomationControlled')

    page = ChromiumPage(addr_or_opts=co)
    page.set.window.max()

    time.sleep(2)

    page.get("https://thuvienphapluat.vn")
    time.sleep(2)

    username_field = page.ele('#usernameTextBox', timeout=3)
    if username_field:
        username_field.input(MY_USERNAME)
        page.ele('#passwordTextBox').input(MY_PASSWORD)
        page.ele('#loginButton').click()

        print("Đang xử lý popup cảnh báo...")
        for _ in range(3):
            try:
                agree_btn = page.ele('tag:button@@text():Đồng ý', timeout=3)
                if agree_btn:
                    agree_btn.click()
                    print("Đã nhấn 'Đồng ý' thành công!")
                    break
            except Exception:
                time.sleep(1)

        time.sleep(3)

    return page


def perform_login(page):
    if page.url != "https://thuvienphapluat.vn":
        page.get("https://thuvienphapluat.vn")
        time.sleep(2)

    username_field = page.ele('#usernameTextBox', timeout=3)
    if username_field:
        print("Đang tiến hành điền thông tin đăng nhập...")
        username_field.input(MY_USERNAME)
        page.ele('#passwordTextBox').input(MY_PASSWORD)
        page.ele('#loginButton').click()

        print("Đang xử lý popup cảnh báo...")
        for _ in range(3):
            try:
                # Tìm nút Đồng ý (chống nháy DOM)
                agree_btn = page.ele('tag:button@@text():Đồng ý', timeout=3)
                if agree_btn:
                    agree_btn.click()
                    print("Đã nhấn 'Đồng ý' thành công!")
                    break
            except Exception:
                time.sleep(1)

        time.sleep(3)
        return True
    else:
        print("Không tìm thấy ô đăng nhập (Có thể đã login sẵn).")
        return False


def handle_anti_bot(page):
    cf_widget = page.ele('@name=cf-turnstile-response', timeout=2)
    if cf_widget:
        print("  [!] Bị chặn bởi Cloudflare, đang tự động click...")
        time.sleep(2)
        try:
            cf_widget.parent().click.at(30, 30)

            is_cleared = page.wait.ele_deleted('@name=cf-turnstile-response', timeout=15)

            if not is_cleared:
                print("  [-] Cloudflare giải quá lâu hoặc tự refresh lại.")
                return False  # Trả về False để vòng lặp bên ngoài F5 làm lại

            print("  [+] Đã vượt qua Cloudflare!")
            time.sleep(1.5)
        except Exception as e:
            print("  [-] Lỗi tương tác Cloudflare:", e)
            return False

    img_captcha = page.ele('xpath://img[@src="/RegistImage.aspx"]', timeout=1)
    if img_captcha or "check.aspx" in page.url:
        print("  [!] Phát hiện CAPTCHA hình ảnh, đang giải mã...")
        try:
            img_captcha = page.ele('xpath://img[@src="/RegistImage.aspx"]', timeout=3)
            if not img_captcha:
                print("  [-] Lỗi: Không bắt được thẻ ảnh Captcha.")
                return False

            img_path = "temp_captcha.png"
            if os.path.exists(img_path):
                os.remove(img_path)

            img_captcha.save(path='.', name=img_path)
            time.sleep(0.5)

            with open(img_path, 'rb') as f:
                captcha_text = ocr.classification(f.read())

            print(f"  [>] AI đoán mã: {captcha_text}")

            input_box = page.ele('#ctl00_Content_txtSecCode')
            input_box.clear()
            input_box.input(captcha_text)
            page.ele('#ctl00_Content_CheckButton').click()

            time.sleep(2)

            if page.ele('#ctl00_Content_txtSecCode', timeout=1):
                print("  [-] AI đoán sai mã Captcha, trang bắt nhập lại.")

                try:
                    page.run_js(
                        'document.querySelector("img[src=\'/RegistImage.aspx\']").src = "/RegistImage.aspx?t=" + new Date().getTime();')
                    time.sleep(1)
                except:
                    pass

                return False

            print("  [+] Đã qua ải Captcha ảnh!")
        except Exception as e:
            print("  [-] Lỗi giải Captcha ảnh:", e)
            return False

    if not page.ele('.content1', timeout=3) and not page.ele('.nqTitle', timeout=1):
        print("  [!] Vẫn chưa thấy dữ liệu thật. Có thể kẹt mạng hoặc dính bot ẩn.")
        return False

    return True

def clean_html_whitespace(html_element):
    if not html_element: return
    for text_node in html_element.find_all(string=True):
        if text_node.parent and text_node.parent.name not in ['pre', 'code']:
            cleaned_text = re.sub(r'\s+', ' ', text_node)
            text_node.replace_with(cleaned_text)


def fetch_all_tabs_html(page, url):
    if page.url != url:
        page.get(url)
        # handle_anti_bot(page)

    if not page.ele('.content1', timeout=3):
        print("  [?] Không thấy nội dung chính, kiểm tra rào cản bot...")
        bot_cleared = False
        for _ in range(3):
            if handle_anti_bot(page):
                bot_cleared = True
                break
            print("  [R] Đang refresh trang để thử lại Bypass...")
            page.refresh()
            time.sleep(2)

        if not bot_cleared:
            raise Exception("BOT_BLOCKED")

    payload = {
        "main_html": page.html,
        "luocdo_html": "",
        "hl_html": "",
        "nd_html": ""
    }

    tabs_config = [
        {"btn_id": "#aLuocDo", "tab_id": "#tab4", "key": "luocdo_html"},
        {"btn_id": "#aLienQuanHL", "tab_id": "#tab5", "key": "hl_html"},
        {"btn_id": "#aLienQuanND", "tab_id": "#tab6", "key": "nd_html"}
    ]

    target_xpath = 'xpath:.//table | .//*[@class="ct"] | .//*[@class="nqTitle"] | .//*[contains(text(), "Không có văn bản")]'

    max_refresh = 3
    refresh_count = 0

    while refresh_count <= max_refresh:
        missing_tabs = False

        for tab in tabs_config:
            if payload[tab["key"]] != "":
                continue

            btn = page.ele(tab["btn_id"], timeout=2)
            if btn:
                print(f"    [*] Đang cào Tab {tab['btn_id']} (Lần thử refresh: {refresh_count})...")
                tab_container = page.ele(tab["tab_id"])

                page.actions.move_to(btn).click()

                data_signal = tab_container.ele(target_xpath, timeout=6)

                if data_signal:
                    html_content = tab_container.html
                    if len(html_content) > 500 or "Không có văn bản" in html_content:
                        payload[tab["key"]] = html_content
                        print(f"    [+] Tab {tab['btn_id']}: Thành công.")

                if payload[tab["key"]] == "":
                    missing_tabs = True
                    print(f"    [!] Tab {tab['btn_id']} vẫn trống.")

        if not missing_tabs:
            break

        refresh_count += 1
        if refresh_count <= max_refresh:
            print(f"    [R] Thiếu dữ liệu, đang refresh trang để thử lại lần {refresh_count}...")
            page.refresh()
            time.sleep(2)

            if not page.ele('.content1', timeout=3):
                print("    [?] Cảm nhận thấy bot sau khi refresh, đang xử lý...")
                if not handle_anti_bot(page):
                    # Giải thất bại thì văng lỗi luôn, không chạy mù quáng
                    raise Exception("BOT_BLOCKED_AFTER_REFRESH")

    payload["main_html"] = page.html
    return payload

def parse_related_documents(html: str, tab_id: str):
    if not html: return []
    soup = BeautifulSoup(html, "html.parser")
    div_tab = soup.find("div", id=tab_id, class_="contentDoc")
    docs = []
    if div_tab:
        related_divs = div_tab.find_all("div", class_=lambda c: c and c.startswith("content-"))
        for div in related_divs:
            number = div.find("div", class_="number").get_text(strip=True) if div.find("div", class_="number") else None
            title_tag = div.find("p", class_="nqTitle")
            link_tag = title_tag.find("a") if title_tag else None
            title = link_tag.get_text(strip=True) if link_tag else None
            href = link_tag.get("href") if link_tag else None
            lawid = title_tag.get("lawid") if title_tag else None

            attributes = {}
            right_col = div.find("div", class_="right-col")
            if right_col:
                for p in right_col.find_all("p"):
                    spans = p.find_all("span")
                    if spans:
                        key = spans[0].get_text(strip=True).replace(":", "")
                        value = p.get_text(strip=True).replace(spans[0].get_text(strip=True), "").strip()
                        if key and value: attributes[key] = value

            docs.append({"number": number, "lawid": lawid, "title": title, "href": href, "attributes": attributes})
    return docs


def parse_luocdo(html: str):
    if not html: return {}
    soup = BeautifulSoup(html, "html.parser")
    luocdo_div = soup.find("div", id="tab4", class_="contentDoc")
    docs_by_category = {}
    if luocdo_div:
        for section in luocdo_div.find_all("div", class_="ct"):
            ghd = section.find_previous_sibling("div", class_="ghd")
            category = ghd.get_text(strip=True) if ghd else "Khác"
            if category == 'Văn bản đang xem': continue

            items = []
            for dgc in section.find_all("div", class_="dgc"):
                link_tag = dgc.find("a")
                href = link_tag.get("href") if link_tag else None

                metadata = {}
                title = None
                tooltip_div = dgc.find("div", class_=lambda c: c and c.startswith("clsTooltip-"))
                if tooltip_div:
                    title_div = tooltip_div.find("div", style=lambda s: s and "background-color: #FFFBF4" in s)
                    if title_div: title = title_div.get_text(strip=True)
                    rows = tooltip_div.find_all("div", style=lambda s: s and "float:left" in s)
                    for i in range(0, len(rows), 2):
                        key = rows[i].get_text(strip=True).replace(":", "")
                        if i + 1 < len(rows):
                            value = rows[i + 1].get_text(strip=True)
                            if key and value: metadata[key] = value

                items.append({"title": title, "href": href, "metadata": metadata})
            docs_by_category[category] = items
    return docs_by_category


def parse_amendments_with_anchor(html: str):
    if not html: return []
    soup = BeautifulSoup(html, "html.parser")
    anchor_map = {}
    div_content = soup.find("div", class_="content1")
    if div_content:
        for a in div_content.find_all("a"):
            atmm = a.get("atmm")
            if atmm and atmm.startswith(".lqhlTootip-"):
                tip_id = atmm.replace(".lqhlTootip-", "")
                anchor_map[tip_id] = a.get_text(strip=True)

    amendments = []
    container = soup.find("div", id="divltrLienQuanHieuLucTungPhan")
    if container:
        for tip_div in container.find_all("div", class_=lambda c: c and c.startswith("lqhlTip-")):
            tip_id = tip_div.get("class")[0].replace("lqhlTip-", "")
            anchor_text = anchor_map.get(tip_id, "")

            bm_title_div = tip_div.find("div", id="bmTitle")
            bm_title = bm_title_div.get_text(strip=True) if bm_title_div else ""

            bm_content_old_div = tip_div.find("div", id="bmContentOld")
            bm_content_old = bm_content_old_div.get_text("\n", strip=True) if bm_content_old_div else ""

            bm_content = ""
            amended_link = ""

            bm_content_div = tip_div.find("div", id="bmContent")
            if bm_content_div:
                a_tag = bm_content_div.find("a", href=True)
                if a_tag:
                    amended_link = a_tag.get("href")
                    if amended_link and amended_link.startswith("/"):
                        amended_link = "https://thuvienphapluat.vn" + amended_link

                    a_tag.extract()

                bm_content = bm_content_div.get_text("\n", strip=True)

            amendments.append({
                "tip_id": tip_id,
                "anchor_text": anchor_text,
                "bm_title": bm_title,
                "original_content": bm_content_old,
                "amended_content": bm_content,
                "amended_link": amended_link
            })

    return amendments

def parse_full_data(html_payload: dict, source_url: str):
    main_html = html_payload["main_html"]
    soup = BeautifulSoup(main_html, 'html.parser')

    toc_items = []
    div_mucluc = soup.find('ul', class_='muclucVB')
    if div_mucluc:
        for li in div_mucluc.find_all('li'):
            a_tag = li.find('a', class_='amuclucvb')
            if a_tag:
                toc_items.append({"title": a_tag.get_text(strip=True), "href": a_tag.get('href')})

    # 2. XỬ LÝ NỘI DUNG VÀ BẢO TỒN MỎ NEO (ANCHOR)
    div_content = soup.find('div', class_='content1')
    if div_content:
        # Xóa comment và các thẻ rác
        for comment in div_content.find_all(string=lambda text: isinstance(text, Comment)): comment.extract()
        for span in div_content.find_all('span'):
            span_text = span.get_text(strip=True)
            if span_text.startswith("VABW") or span_text.startswith("LdAB"): span.extract()
        for table in div_content.find_all('table'):
            table_text = " ".join(table.stripped_strings).upper()
            if ("CỘNG HÒA" in table_text and "ĐỘC LẬP" in table_text) or "NƠI NHẬN:" in table_text: table.extract()

        anchor_ids = [item['href'].replace('#', '') for item in toc_items if
                      item['href'] and item['href'].startswith('#')]

        for anchor_id in anchor_ids:
            target_element = div_content.find(attrs={"name": anchor_id}) or div_content.find(attrs={"id": anchor_id})
            if target_element:
                marker = soup.new_string(f"\n\n[[ANCHOR:{anchor_id}]]\n\n")
                target_element.insert_before(marker)

        injected_amends = set()

        for a_tag in div_content.find_all('a'):
            atmm = a_tag.get('atmm')
            if atmm and atmm.startswith('.lqhlTootip-'):
                tip_id = atmm.replace('.lqhlTootip-', '')

                if tip_id not in injected_amends:
                    marker_amend = soup.new_string(f"\n[[AMENDMENT:{tip_id}]]\n")
                    a_tag.insert_before(marker_amend)
                    injected_amends.add(tip_id)

        for a_tag in div_content.find_all('a'):
            a_tag.unwrap()

        clean_html_whitespace(div_content)
        raw_md = md(str(div_content), heading_style="ATX")

        raw_md = raw_md.replace(r"\_", "_")

        raw_md = re.sub(r'(\*\*(?:Chương|Phần|Mục|Phụ lục)\s+[A-Za-z0-9IVX]+)\*\*\s+\*\*', r'\1 - ', raw_md,
                        flags=re.IGNORECASE)
        raw_md = re.sub(r'(\*\*Điều\s+\d+\.?)\*\*\s+\*\*', r'\1 ', raw_md, flags=re.IGNORECASE)

        raw_md = re.sub(r'VABWAFAATABf[A-Za-z0-9+/=]+', '', raw_md)
        raw_md = re.sub(r'LdABoAHUAdgBp[A-Za-z0-9+/=]+', '', raw_md)
        main_text = re.sub(r'\n{3,}', '\n\n', raw_md).strip()
    else:
        main_text = ""

    summary_div = soup.find('div', id='tab-1', class_='contentDoc')
    title, metadata, content_summary_text = "", {}, ""

    if summary_div:
        title_tag = summary_div.find('h1')
        title = title_tag.get_text(strip=True) if title_tag else ""

        table = summary_div.find('table')
        if table:
            rows = table.find_all("tr")
            for row in rows:
                cells = row.find_all("td")
                for idx, cell in enumerate(cells):
                    b_tag = cell.find("b")
                    if b_tag:
                        key = b_tag.get_text(strip=True).replace(":", "")
                        if idx + 1 < len(cells):
                            value = cells[idx + 1].get_text(strip=True)
                            if key == "Tình trạng" and value == "Đã biết":
                                print("  [!] Cảnh báo: Tài khoản hết hạn (Tình trạng: Đã biết).")
                                raise Exception("SESSION_EXPIRED")
                            if key and value: metadata[key] = value

        content_summary_div = summary_div.find('div', class_='Tomtatvanban')
        if content_summary_div:
            clean_html_whitespace(content_summary_div)
            raw_summary_md = md(str(content_summary_div), heading_style="ATX")
            content_summary_text = re.sub(r'\n{3,}', '\n\n', raw_summary_md).strip()

    return {
        "source_url": source_url,
        "title": title,
        "metadata": metadata,
        "summary_content": content_summary_text,
        "mucluc": toc_items,
        "main_content": main_text,
        "amendments": parse_amendments_with_anchor(main_html),
        "lienquan_hieuluc_list": parse_related_documents(html_payload["hl_html"], "tab5"),
        "related_noidung_list": parse_related_documents(html_payload["nd_html"], "tab6"),
        "luoc_do_list": parse_luocdo(html_payload["luocdo_html"])
    }


def scrape_full_database():
    page = init_browser()
    page.set.retry_times(1)

    urls_to_crawl_next = pd.read_csv("urls_to_crawl_next.csv")["url"].to_list()

    try:
        for link in urls_to_crawl_next:
            full_url = link if link.startswith("http") else BASE_URL + link
            filename = sanitize_filename(full_url) + ".json"
            filepath = os.path.join(OUTPUT_DIR, filename)

            if os.path.exists(filepath): continue

            retry_crawl = True
            max_retries = 3
            attempt = 0

            while retry_crawl and attempt < max_retries:
                try:
                    payload = fetch_all_tabs_html(page, full_url)
                    data = parse_full_data(payload, full_url)

                    data.update({
                        "lienquan_hieuluc_list": parse_related_documents(payload["hl_html"], "tab5"),
                        "related_noidung_list": parse_related_documents(payload["nd_html"], "tab6"),
                        "luoc_do_list": parse_luocdo(payload["luocdo_html"]),
                        "amendments": parse_amendments_with_anchor(payload["main_html"])
                    })

                    # Lưu file JSON
                    with open(filepath, "w", encoding="utf-8") as f:
                        json.dump(data, f, ensure_ascii=False)

                    print(f"  [OK] {filename}")
                    retry_crawl = False

                except Exception as e:
                    error_msg = str(e)
                    attempt += 1

                    if "SESSION_EXPIRED" in error_msg:
                        print(f"  [R] Lần {attempt}: Tài khoản hết hạn! Đang đăng nhập lại...")
                        perform_login(page)
                    elif "The page is refreshed" in error_msg:
                        print(f"  [R] Lần {attempt}: Trang đang refresh đột ngột, chờ 3s rồi cào lại...")
                        time.sleep(3)
                        page.wait.doc_loaded(timeout=5)
                    else:
                        print(f"  [Lỗi] {full_url}: {error_msg}")
                        time.sleep(2)

                    if attempt >= max_retries:
                        print(f"  [!] Bỏ qua {full_url} sau {max_retries} lần thử thất bại.")
                        retry_crawl = False

            time.sleep(random.uniform(1.0, 2.0))

        time.sleep(1.5)

    finally:
        page.quit()


if __name__ == "__main__":
    scrape_full_database()