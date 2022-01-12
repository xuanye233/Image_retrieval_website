from datetime import datetime
import time
from urllib.parse import urlparse, parse_qs

import alipay
from flask import Flask, request, url_for, redirect

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def pay():
    if request.method == 'POST':
        newpay = alipay.AliPay(
            appid="2016101600703811",  # 设置签约的appid
            app_notify_url="http://112.74.55.3/notify/",  # 异步支付通知url
            app_private_key_path=alipay.app_private_key_path,  # 设置应用私钥
            alipay_public_key_path=alipay.alipay_public_key_path,  # 支付宝的公钥，验证支付宝回传消息使用，不是你自己的公钥,
            debug=True,  # 默认False,            # 设置是否是沙箱环境，True是沙箱环境
            return_url="http://127.0.0.1:5000/result/",  # 同步支付通知url,在这个页面可以展示给用户看，只有付款成功后才会跳转
        )
        url = newpay.direct_pay(
            subject="测试订单",  # 订单名称
            # 订单号生成，一般是当前时间(精确到秒)+用户ID+随机数
            out_trade_no=str(time.strftime('%Y%m%d%H%M%S', time.localtime(time.time())))+ str(time.time()).replace('.', '')[-7:],  # 订单号
            total_amount=0.01,  # 支付金额
            return_url="http://127.0.0.1:5000/result/",  # 支付成功后，跳转url
        )
        re_url = "https://openapi.alipaydev.com/gateway.do?{data}".format(data=url)
        return redirect(re_url)

    return '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Title</title>
    </head>
    <body>
        <form action="/" method="POST">
            <input type="submit" value="去支付" />
        </form>
    <script></script>
    </body>
    </html>
    '''


@app.route('/result/')
def result():

    return'''
        <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Title</title>
    </head>
    <body>
        <form action="/" method="POST">
            支付成功
        </form>
    <script></script>
    </body>
    </html>
    '''

if __name__ == '__main__':
    app.run()

