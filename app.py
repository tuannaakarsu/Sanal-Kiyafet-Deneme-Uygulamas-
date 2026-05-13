from oneri import oneri_uret
import os
import uuid
import datetime
from flask_cors import CORS
from anakod import run_virtual_tryon_auto
from flask import Flask ,request, jsonify , send_file , send_from_directory , make_response

app = Flask(__name__ , static_folder='tumkatalog')
CORS(app)

@app.route('/vton', methods=['POST'])
def vton():
    ####### json dosyası gelmiş ise
    if request.is_json:
        try:
            data = request.get_json()
            kullanici_verisi = data.get("clothes")

            if not kullanici_verisi:
                return jsonify({"error": "clothes alanı eksik"}), 400

            ngrok_url = "https://5572-81-214-127-142.ngrok-free.app"
            oneriler = oneri_uret(kullanici_verisi,ngrok_url)
            return jsonify({"oneriler": oneriler}), 200

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    elif "user_image" and "clothing_image" in request.files:
        try:
            user_file = request.files["user_image"]
            clothes_file = request.files["clothing_image"]

            user_path = "temp/user.jpeg"
            clothes_path = "temp/clothes.jpeg"

            user_file.save(user_path)
            clothes_file.save(clothes_path)

            unique_id = str(uuid.uuid4())
            result_path = f"temp/result_{unique_id}.jpeg"

            run_virtual_tryon_auto(
                person_img_path=user_path,
                clothes_img_path=clothes_path,
                unet_model_path="models/unet_model_2.pth",
                output_path= result_path
            )

            response = make_response(send_file(result_path, mimetype='image/jpeg'))

            response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
            response.headers['Pragma'] = 'no-cache'
            response.headers['Expires'] = '0'
            response.headers['X-Process-Time'] = datetime.datetime.now().isoformat()
            response.headers['X-Result-File'] = os.path.basename(result_path)

            return response

        except Exception as e:
            print("HATA:", str(e))
            return jsonify({"error": str(e)}), 500

    else:
        return jsonify({"error": "Geçersiz istek. JSON ya da iki resim dosyası bekleniyor."}), 400
@app.route('/tumkatalog/<path:filename>')
def serve_images(filename):
    return send_from_directory('tumkatalog', filename)

if __name__ == '__main__':
    app.run(debug=True)



