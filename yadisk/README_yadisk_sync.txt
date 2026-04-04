Файлы:
- yadisk_simple_sync_common.py
- yadisk_simple_upload.py
- yadisk_simple_download.py

Установка:
    pip install requests

Загрузка:
    python yadisk_simple_upload.py --local "/path/to/local" --remote-root "disk:/backup" --token "TOKEN"

Скачивание:
    python yadisk_simple_download.py --local "/path/to/local" --remote-root "disk:/backup" --token "TOKEN"

Чтобы перезаписывать обычные файлы с тем же именем, но другим содержимым:
    --overwrite-different-size

Правила для сравнения:
- обычные файлы считаются совпавшими, если совпадает md5;
- размер больше не используется как критерий совпадения;
- если удалённый md5 недоступен, файл считается несовпавшим.

Правила для plots:
- локальная папка с именем plots загружается как plots.zip;
- на Яндекс Диске такие zip-файлы хранятся как *.zipfast;
- если всё кроме plots.zip в соответствующем subtree уже идеально синхронизировано и удалённый plots.zip существует, upload считает plots синхронизированным без локальной сборки zip;
- иначе upload архивирует локальный plots/, сравнивает md5 с удалённым plots.zip и при необходимости загружает его;
- download никогда не архивирует локальный plots/ для сравнения с удалённым plots.zip;
- download всегда обновляет локальный plots/ из удалённого plots.zip: скачивает архив и распаковывает его;
- --local не должен указывать прямо на папку plots/.
