"""
Главное окно приложения.
"""

import json
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox

from loguru import logger

from desktop.ui.components import StatusBar, ChatHistoryWidget


class MainWindow:
    """Главное окно приложения."""
    
    def __init__(self, root: tk.Tk, api_client, config_manager):
        """
        Инициализация главного окна.
        
        Args:
            root: Корневое окно tkinter
            api_client: Клиент API Scrivener
            config_manager: Менеджер конфигурации
        """
        self.root = root
        self.api_client = api_client
        self.config_manager = config_manager
        
        self.root.title("Smart RAG - Desktop App")
        self.root.geometry("1000x700")
        
        # Создаем Notebook для вкладок
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Создаем вкладки
        self.chat_frame = ttk.Frame(self.notebook)
        self.rag_frame = ttk.Frame(self.notebook)
        self.collections_frame = ttk.Frame(self.notebook)
        self.settings_frame = ttk.Frame(self.notebook)
        self.user_frame = ttk.Frame(self.notebook)
        
        self.notebook.add(self.chat_frame, text="Чат")
        self.notebook.add(self.rag_frame, text="RAG управление")
        self.notebook.add(self.collections_frame, text="Коллекции")
        self.notebook.add(self.settings_frame, text="Настройки")
        self.notebook.add(self.user_frame, text="Пользователь")
        
        # Строка статуса
        self.status_bar = StatusBar(self.root)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Переменные для хранения данных
        self.chat_history_irv_id = None
        self.current_irv_id = None
        
        # Инициализируем вкладки
        self._init_chat_tab()
        self._init_rag_tab()
        self._init_collections_tab()
        self._init_settings_tab()
        self._init_user_tab()
    
    def set_status(self, message: str, color: str = "black"):
        """Устанавливает статус в строке статуса."""
        self.status_bar.set_status(message, color)
    
    def _init_chat_tab(self):
        """Инициализация вкладки 'Чат'."""
        # Левая панель - настройки
        left_panel = ttk.Frame(self.chat_frame)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        # Режимы
        modes_frame = ttk.LabelFrame(left_panel, text="Режимы", padding=10)
        modes_frame.pack(fill=tk.X, pady=5)
        
        self.internet_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(modes_frame, text="Использовать интернет", variable=self.internet_var).pack(anchor=tk.W)
        
        self.knowledge_base_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(modes_frame, text="Использовать базу знаний", variable=self.knowledge_base_var).pack(anchor=tk.W)
        
        # ИО
        irv_frame = ttk.LabelFrame(left_panel, text="Информационный объект", padding=10)
        irv_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(irv_frame, text="IRV ID (опционально):").pack(anchor=tk.W)
        self.irv_id_var = tk.StringVar()
        ttk.Entry(irv_frame, textvariable=self.irv_id_var, width=30).pack(fill=tk.X, pady=2)
        
        ttk.Label(irv_frame, text="Chat History IRV ID:").pack(anchor=tk.W, pady=(5, 0))
        self.chat_history_irv_id_var = tk.StringVar()
        ttk.Entry(irv_frame, textvariable=self.chat_history_irv_id_var, width=30).pack(fill=tk.X, pady=2)
        
        # Правая панель - чат
        right_panel = ttk.Frame(self.chat_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # История чата
        history_label = ttk.Label(right_panel, text="История диалога:")
        history_label.pack(anchor=tk.W)
        
        self.chat_history = ChatHistoryWidget(right_panel, height=20)
        self.chat_history.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Поле ввода
        input_frame = ttk.Frame(right_panel)
        input_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(input_frame, text="Сообщение:").pack(anchor=tk.W)
        self.message_entry = scrolledtext.ScrolledText(input_frame, height=3, wrap=tk.WORD)
        self.message_entry.pack(fill=tk.X, pady=2)
        
        # Кнопки
        buttons_frame = ttk.Frame(right_panel)
        buttons_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(buttons_frame, text="Отправить", command=self._send_message).pack(side=tk.LEFT, padx=2)
        ttk.Button(buttons_frame, text="Очистить историю", command=self._clear_chat_history).pack(side=tk.LEFT, padx=2)
        ttk.Button(buttons_frame, text="Сохранить историю", command=self._save_chat_history).pack(side=tk.LEFT, padx=2)
    
    def _send_message(self):
        """Отправка сообщения."""
        message = self.message_entry.get(1.0, tk.END).strip()
        if not message:
            messagebox.showwarning("Предупреждение", "Введите сообщение")
            return
        
        # Добавляем сообщение пользователя в историю
        self.chat_history.add_message("user", message)
        self.message_entry.delete(1.0, tk.END)
        
        # Получаем настройки LLM из конфигурации
        llm_config = self.config_manager.get("llm", {})
        
        # Формируем запрос
        request_data = {
            "current_message": message,
            "chat_history_irv_id": self.chat_history_irv_id_var.get().strip() or None,
            "irv_id": self.irv_id_var.get().strip() or None,
            "temperature": llm_config.get("temperature", 0.2),
            "max_tokens": llm_config.get("max_tokens", 8000),
            "llm_api_key": llm_config.get("api_key", ""),
            "llm_url": llm_config.get("url", ""),
            "llm_model_name": llm_config.get("model", "gpt-4o-mini"),
            "llm_auth_type": llm_config.get("llm_auth_type", 0),
            "internet": self.internet_var.get(),
            "knowledge_base": self.knowledge_base_var.get(),
        }
        
        # Добавляем опциональные параметры
        config = self.config_manager.config
        
        # Системный промпт передается только если не используется интернет и база знаний
        if not request_data["internet"] and not request_data["knowledge_base"]:
            system_prompt = llm_config.get("system_prompt", "")
            if system_prompt:
                request_data["system_prompt"] = system_prompt
        
        if config.get("embeddings", {}).get("api_key"):
            request_data["embed_api_key"] = config["embeddings"]["api_key"]
            request_data["embed_url"] = config["embeddings"].get("url")
            request_data["embed_model_name"] = config["embeddings"].get("model")
        
        if config.get("qdrant", {}).get("url"):
            request_data["vdb_url"] = config["qdrant"]["url"]
        
        if config.get("search", {}).get("api_key"):
            request_data["search_api_key"] = config["search"]["api_key"]
            request_data["search_url"] = config["search"].get("url")
        
        self.set_status("Отправка запроса...", "blue")
        self.root.update()
        
        # Отправляем запрос
        result = self.api_client.generate_response(request_data)
        
        if result.get("success"):
            content = result.get("data", {}).get("content", "")
            if isinstance(content, dict):
                content = json.dumps(content, ensure_ascii=False, indent=2)
            self.chat_history.add_message("assistant", content)
            self.set_status("Ответ получен", "green")
        else:
            error_msg = result.get("detail") or result.get("error", "Неизвестная ошибка")
            self.chat_history.add_message("error", f"Ошибка: {error_msg}")
            self.set_status(f"Ошибка: {error_msg}", "red")
            messagebox.showerror("Ошибка", error_msg)
    
    def _clear_chat_history(self):
        """Очистка истории чата."""
        self.chat_history.clear()
        self.chat_history_irv_id_var.set("")
        self.set_status("История очищена", "green")
    
    def _save_chat_history(self):
        """Сохранение истории чата."""
        # Простое сохранение в файл (в реальности должно сохраняться в эмулятор КФО)
        from tkinter import filedialog
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filename:
            # Извлекаем сообщения из истории (упрощенная версия)
            content = self.chat_history.get(1.0, tk.END)
            # В реальности нужно парсить сообщения и сохранять в формате chat_history.json
            with open(filename, "w", encoding="utf-8") as f:
                f.write(content)
            self.set_status(f"История сохранена в {filename}", "green")
    
    def _init_rag_tab(self):
        """Инициализация вкладки 'RAG управление'."""
        # Создаем Notebook для разделения на этапы
        notebook = ttk.Notebook(self.rag_frame)
        notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Вкладка 1: Загрузка в хранилище
        storage_frame = ttk.Frame(notebook)
        notebook.add(storage_frame, text="Загрузка в хранилище")
        self._init_storage_tab(storage_frame)
        
        # Вкладка 2: Добавление в RAG
        rag_add_frame = ttk.Frame(notebook)
        notebook.add(rag_add_frame, text="Добавление в RAG")
        self._init_rag_add_tab(rag_add_frame)
        
        # Вкладка 3: Управление RAG
        rag_manage_frame = ttk.Frame(notebook)
        notebook.add(rag_manage_frame, text="Управление RAG")
        self._init_rag_manage_tab(rag_manage_frame)
    
    def _init_storage_tab(self, parent):
        """Инициализация вкладки загрузки в хранилище."""
        # Верхняя панель - IRV ID
        top_panel = ttk.Frame(parent)
        top_panel.pack(fill=tk.X, padx=5, pady=5)
        
        config_frame = ttk.LabelFrame(top_panel, text="Информационный объект", padding=10)
        config_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(config_frame, text="IRV ID:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.storage_irv_id_var = tk.StringVar()
        ttk.Entry(config_frame, textvariable=self.storage_irv_id_var, width=50).grid(row=0, column=1, pady=2, padx=5)
        ttk.Button(config_frame, text="Создать новый", command=self._create_new_irv).grid(row=0, column=2, padx=5)
        
        # Средняя панель - файлы для загрузки
        middle_panel = ttk.Frame(parent)
        middle_panel.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        files_frame = ttk.LabelFrame(middle_panel, text="Файлы для загрузки", padding=10)
        files_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Список файлов
        list_frame = ttk.Frame(files_frame)
        list_frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(list_frame, text="Выбранные файлы:").pack(anchor=tk.W)
        self.storage_files_listbox = tk.Listbox(list_frame, height=10)
        self.storage_files_listbox.pack(fill=tk.BOTH, expand=True, pady=5)
        
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.storage_files_listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.storage_files_listbox.config(yscrollcommand=scrollbar.set)
        
        self.storage_selected_files = []
        
        # Кнопки управления файлами
        files_buttons = ttk.Frame(files_frame)
        files_buttons.pack(fill=tk.X, pady=5)
        
        ttk.Button(files_buttons, text="Выбрать файлы", command=self._select_storage_files).pack(side=tk.LEFT, padx=2)
        ttk.Button(files_buttons, text="Удалить выбранный", command=self._remove_selected_storage_file).pack(side=tk.LEFT, padx=2)
        ttk.Button(files_buttons, text="Очистить список", command=self._clear_storage_files_list).pack(side=tk.LEFT, padx=2)
        ttk.Button(files_buttons, text="Загрузить в хранилище", command=self._upload_to_storage).pack(side=tk.LEFT, padx=2)
        
        # Нижняя панель - результаты
        bottom_panel = ttk.Frame(parent)
        bottom_panel.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        results_frame = ttk.LabelFrame(bottom_panel, text="Результаты", padding=10)
        results_frame.pack(fill=tk.BOTH, expand=True)
        
        self.storage_results = scrolledtext.ScrolledText(results_frame, height=8, wrap=tk.WORD, state=tk.DISABLED)
        self.storage_results.pack(fill=tk.BOTH, expand=True)
    
    def _init_rag_add_tab(self, parent):
        """Инициализация вкладки добавления в RAG."""
        # Верхняя панель - IRV ID
        top_panel = ttk.Frame(parent)
        top_panel.pack(fill=tk.X, padx=5, pady=5)
        
        config_frame = ttk.LabelFrame(top_panel, text="Информационный объект", padding=10)
        config_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(config_frame, text="IRV ID:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.rag_add_irv_id_var = tk.StringVar()
        ttk.Entry(config_frame, textvariable=self.rag_add_irv_id_var, width=50).grid(row=0, column=1, pady=2, padx=5)
        ttk.Button(config_frame, text="Загрузить список файлов", command=self._load_storage_files).grid(row=0, column=2, padx=5)
        
        # Средняя панель - файлы из хранилища
        middle_panel = ttk.Frame(parent)
        middle_panel.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        files_frame = ttk.LabelFrame(middle_panel, text="Файлы в хранилище", padding=10)
        files_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Список файлов из хранилища
        list_frame = ttk.Frame(files_frame)
        list_frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(list_frame, text="Файлы в хранилище:").pack(anchor=tk.W)
        self.storage_files_listbox_rag = tk.Listbox(list_frame, height=10)
        self.storage_files_listbox_rag.pack(fill=tk.BOTH, expand=True, pady=5)
        
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.storage_files_listbox_rag.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.storage_files_listbox_rag.config(yscrollcommand=scrollbar.set)
        
        self.storage_files_data = []  # Список файлов из хранилища
        
        # Кнопки операций
        buttons_frame = ttk.Frame(middle_panel)
        buttons_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(buttons_frame, text="Добавить файлы в RAG", command=self._add_all_storage_files_to_rag).pack(side=tk.LEFT, padx=2)
        
        # Нижняя панель - результаты
        bottom_panel = ttk.Frame(parent)
        bottom_panel.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        results_frame = ttk.LabelFrame(bottom_panel, text="Результаты", padding=10)
        results_frame.pack(fill=tk.BOTH, expand=True)
        
        self.rag_add_results = scrolledtext.ScrolledText(results_frame, height=8, wrap=tk.WORD, state=tk.DISABLED)
        self.rag_add_results.pack(fill=tk.BOTH, expand=True)
    
    def _init_rag_manage_tab(self, parent):
        """Инициализация вкладки управления RAG."""
        # Верхняя панель - IRV ID
        top_panel = ttk.Frame(parent)
        top_panel.pack(fill=tk.X, padx=5, pady=5)
        
        config_frame = ttk.LabelFrame(top_panel, text="Информационный объект", padding=10)
        config_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(config_frame, text="IRV ID:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.rag_manage_irv_id_var = tk.StringVar()
        ttk.Entry(config_frame, textvariable=self.rag_manage_irv_id_var, width=50).grid(row=0, column=1, pady=2, padx=5)
        
        # Кнопки операций RAG
        buttons_frame = ttk.Frame(parent)
        buttons_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(buttons_frame, text="Удалить из RAG", command=self._remove_from_rag).pack(side=tk.LEFT, padx=2)
        ttk.Button(buttons_frame, text="Информация о файле", command=self._get_file_info).pack(side=tk.LEFT, padx=2)
        
        # Нижняя панель - результаты
        bottom_panel = ttk.Frame(parent)
        bottom_panel.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        results_frame = ttk.LabelFrame(bottom_panel, text="Результаты", padding=10)
        results_frame.pack(fill=tk.BOTH, expand=True)
        
        self.rag_manage_results = scrolledtext.ScrolledText(results_frame, height=8, wrap=tk.WORD, state=tk.DISABLED)
        self.rag_manage_results.pack(fill=tk.BOTH, expand=True)
    
    # Методы для работы с хранилищем
    def _select_storage_files(self):
        """Выбор файлов для загрузки в хранилище."""
        from tkinter import filedialog
        files = filedialog.askopenfilenames(
            title="Выберите файлы",
            filetypes=[("Документы", "*.docx *.txt"), ("Все файлы", "*.*")]
        )
        if files:
            for file_path in files:
                if file_path not in self.storage_selected_files:
                    self.storage_selected_files.append(file_path)
                    self.storage_files_listbox.insert(tk.END, file_path)
    
    def _remove_selected_storage_file(self):
        """Удаление выбранного файла из списка."""
        selection = self.storage_files_listbox.curselection()
        if selection:
            index = selection[0]
            self.storage_files_listbox.delete(index)
            del self.storage_selected_files[index]
    
    def _clear_storage_files_list(self):
        """Очистка списка файлов."""
        self.storage_files_listbox.delete(0, tk.END)
        self.storage_selected_files.clear()
    
    def _create_new_irv(self):
        """Создание нового IRV ID."""
        import uuid
        new_irv_id = str(uuid.uuid4())
        self.storage_irv_id_var.set(new_irv_id)
        self.set_status(f"Создан новый IRV ID: {new_irv_id}", "green")
    
    def _upload_to_storage(self):
        """Загрузка файлов в хранилище КФО."""
        if not self.storage_selected_files:
            messagebox.showwarning("Предупреждение", "Выберите файлы для загрузки")
            return
        
        # Получаем или создаем IRV ID
        irv_id = self.storage_irv_id_var.get().strip()
        if not irv_id:
            messagebox.showwarning("Предупреждение", "Введите или создайте IRV ID")
            return
        
        # Очищаем поле результатов
        self.storage_results.config(state=tk.NORMAL)
        self.storage_results.delete(1.0, tk.END)
        self.storage_results.config(state=tk.DISABLED)
        
        # Загружаем файлы в эмулятор КФО
        self.set_status("Загрузка файлов в хранилище КФО...", "blue")
        self.root.update()
        
        import httpx
        import uuid as uuid_lib
        from pathlib import Path
        
        cfx_url = self.api_client.cfx_emulator_url
        
        try:
            # Получаем имя первого файла для использования как имя ИО
            first_file_name = ""
            if self.storage_selected_files:
                first_file_path = Path(self.storage_selected_files[0])
                if first_file_path.exists():
                    first_file_name = first_file_path.name  # Имя файла с расширением
            
            # Создаем ИО если его еще нет
            try:
                with httpx.Client(timeout=10.0) as client:
                    response = client.get(
                        f"{cfx_url}/siu-star/services/api/irv/{irv_id}",
                        cookies={"JSESSIONID": "debug"}
                    )
                    if response.status_code == 404:
                        # Создаем новый ИО с именем первого файла
                        folder_id = "root"
                        irv_name = first_file_name if first_file_name else "ИО для файлов"
                        irv_data = {
                            "name": irv_name,
                            "description": f"ИО создан для загрузки файлов",
                            "nauId": None,
                        }
                        response = client.post(
                            f"{cfx_url}/siu-star/services/api/folder/{folder_id}/irvs",
                            json=irv_data,
                            cookies={"JSESSIONID": "debug"}
                        )
                        if response.status_code == 200:
                            created_data = response.json()
                            irv_id = created_data.get("irv_id", irv_id)
                            self.storage_irv_id_var.set(irv_id)
            except Exception as e:
                logger.warning(f"Ошибка при создании/проверке ИО: {e}")
            
            uploaded_files = []
            # Загружаем файлы
            for file_path in self.storage_selected_files:
                file_path_obj = Path(file_path)
                if not file_path_obj.exists():
                    continue
                
                file_content = file_path_obj.read_bytes()
                irvf_id = str(uuid_lib.uuid4())
                file_name = file_path_obj.name
                
                # Загружаем файл через API эмулятора
                with httpx.Client(timeout=60.0) as client:
                    import hashlib
                    from urllib.parse import quote
                    crc = hashlib.md5(file_content).hexdigest()
                    url = f"{cfx_url}/siu-star/services/api/file/{irvf_id}/write?fileName={quote(file_name)}&crc={crc}&irvId={quote(irv_id)}"
                    response = client.post(
                        url,
                        content=file_content,
                        cookies={"JSESSIONID": "debug"},
                        headers={"Content-Type": "application/octet-stream"}
                    )
                    if response.status_code != 200:
                        logger.error(f"Ошибка загрузки файла {file_name}: {response.text}")
                        raise Exception(f"Не удалось загрузить файл {file_name}: {response.text}")
                    uploaded_files.append(file_name)
            
            self.set_status(f"Файлы загружены в хранилище (IRV ID: {irv_id})", "green")
            
            # Выводим результат
            self.storage_results.config(state=tk.NORMAL)
            self.storage_results.insert(tk.END, f"IRV ID: {irv_id}\n\n")
            self.storage_results.insert(tk.END, f"Загружено файлов: {len(uploaded_files)}\n\n")
            for file_name in uploaded_files:
                self.storage_results.insert(tk.END, f"- {file_name}\n")
            self.storage_results.config(state=tk.DISABLED)
            
            # Очищаем список выбранных файлов
            self._clear_storage_files_list()
            
        except Exception as e:
            logger.exception(f"Ошибка при загрузке файлов: {e}")
            self.storage_results.config(state=tk.NORMAL)
            self.storage_results.insert(tk.END, f"Ошибка: {str(e)}")
            self.storage_results.config(state=tk.DISABLED)
            messagebox.showerror("Ошибка", f"Не удалось загрузить файлы в хранилище: {e}")
    
    def _load_storage_files(self):
        """Загрузка списка файлов из хранилища."""
        irv_id = self.rag_add_irv_id_var.get().strip()
        if not irv_id:
            messagebox.showwarning("Предупреждение", "Введите IRV ID")
            return
        
        # Очищаем поле результатов
        self.rag_add_results.config(state=tk.NORMAL)
        self.rag_add_results.delete(1.0, tk.END)
        self.rag_add_results.config(state=tk.DISABLED)
        
        self.set_status("Загрузка списка файлов из хранилища...", "blue")
        self.root.update()
        
        import httpx
        cfx_url = self.api_client.cfx_emulator_url
        
        try:
            with httpx.Client(timeout=10.0) as client:
                response = client.get(
                    f"{cfx_url}/siu-star/services/api/irv/{irv_id}/files",
                    cookies={"JSESSIONID": "debug"}
                )
                response.raise_for_status()
                files_data = response.json()
                
                # Очищаем список
                self.storage_files_listbox_rag.delete(0, tk.END)
                self.storage_files_data = []
                
                if isinstance(files_data, list):
                    for file_info in files_data:
                        file_name = file_info.get("name", file_info.get("fileName", "неизвестно"))
                        irvf_id = file_info.get("irvfId", "")
                        self.storage_files_listbox_rag.insert(tk.END, file_name)
                        self.storage_files_data.append({
                            "irvf_id": irvf_id,
                            "name": file_name,
                            "irv_id": irv_id
                        })
                
                self.set_status(f"Загружено файлов: {len(self.storage_files_data)}", "green")
                
                # Выводим результат
                self.rag_add_results.config(state=tk.NORMAL)
                self.rag_add_results.insert(tk.END, f"IRV ID: {irv_id}\n\n")
                self.rag_add_results.insert(tk.END, f"Найдено файлов: {len(self.storage_files_data)}\n\n")
                for file_data in self.storage_files_data:
                    self.rag_add_results.insert(tk.END, f"- {file_data['name']}\n")
                self.rag_add_results.config(state=tk.DISABLED)
                
        except Exception as e:
            logger.exception(f"Ошибка при загрузке списка файлов: {e}")
            self.rag_add_results.config(state=tk.NORMAL)
            self.rag_add_results.insert(tk.END, f"Ошибка: {str(e)}")
            self.rag_add_results.config(state=tk.DISABLED)
            messagebox.showerror("Ошибка", f"Не удалось загрузить список файлов: {e}")
    
    def _add_all_storage_files_to_rag(self):
        """Добавление всех файлов из хранилища в RAG."""
        if not self.storage_files_data:
            messagebox.showwarning("Предупреждение", "Сначала загрузите список файлов")
            return
        
        irv_id = self.rag_add_irv_id_var.get().strip()
        if not irv_id:
            messagebox.showwarning("Предупреждение", "Введите IRV ID")
            return
        
        self._add_files_to_rag_from_storage(irv_id, self.storage_files_data)
    
    def _add_files_to_rag_from_storage(self, irv_id: str, files_data: list):
        """Добавление файлов из хранилища в RAG."""
        # Очищаем поле результатов
        self.rag_add_results.config(state=tk.NORMAL)
        self.rag_add_results.delete(1.0, tk.END)
        self.rag_add_results.config(state=tk.DISABLED)
        
        self.set_status("Добавление файлов в RAG...", "blue")
        self.root.update()
        
        # Получаем настройки из конфигурации
        qdrant_config = self.config_manager.get("qdrant", {})
        embed_config = self.config_manager.get("embeddings", {})
        
        # Формируем запрос
        request_data = {
            "vdb_url": qdrant_config.get("url", "http://localhost:6333"),
            "irv_id": irv_id,
            "action": "add",
        }
        
        # Добавляем настройки эмбеддера (если они есть в конфигурации)
        if embed_config.get("api_key"):
            request_data["embed_api_key"] = embed_config.get("api_key")
        if embed_config.get("url"):
            request_data["embed_url"] = embed_config.get("url")
        if embed_config.get("model"):
            request_data["embed_model_name"] = embed_config.get("model")
        
        result = self.api_client.manage_rag_files(request_data)
        
        # Выводим полный ответ от сервера (как получает КФО)
        self.rag_add_results.config(state=tk.NORMAL)
        if result.get("success"):
            # Показываем весь ответ от сервера (result["data"] - это полный JSON ответ)
            server_response = result.get("data", {})
            text = json.dumps(server_response, ensure_ascii=False, indent=2)
            self.rag_add_results.insert(tk.END, text)
            self.set_status(f"Файлы добавлены в RAG (IRV ID: {irv_id})", "green")
        else:
            # При ошибке показываем полный ответ от сервера
            server_response = result.get("data", result)
            text = json.dumps(server_response, ensure_ascii=False, indent=2)
            self.rag_add_results.insert(tk.END, text)
            error_msg = self._format_error_message(result)
            self.set_status(f"Ошибка: {error_msg}", "red")
            messagebox.showerror("Ошибка", error_msg)
        self.rag_add_results.config(state=tk.DISABLED)
    
    def _format_error_message(self, result: dict) -> str:
        """Форматирование сообщения об ошибке для пользователя."""
        error = result.get("error", "Неизвестная ошибка")
        detail = result.get("detail", "")
        code = result.get("code", "")
        
        # Обработка ошибок конфигурации эмбеддингов
        if code == "missing_embed_api_key":
            return f"{error}\n\n{detail}\n\nДля исправления укажите параметр embed_api_key в запросе или установите переменную окружения GIGACHAT_AUTH_KEY."
        
        if code == "empty_embed_api_key":
            return f"{error}\n\n{detail}\n\nУбедитесь, что параметр embed_api_key в запросе содержит корректный API ключ."
        
        # Обработка ошибок соединения
        if code == "connection_error":
            error_str = str(detail) if detail else str(error)
            
            # Определяем, к какому сервису пытались подключиться
            if "gigachat" in error_str.lower() or "embeddings" in error_str.lower() or "embed" in error_str.lower():
                return f"Не удалось подключиться к сервису эмбеддингов.\n\nДетали: {detail if detail else error}\n\nПроверьте:\n" \
                       f"1. Правильность параметра embed_url в запросе\n" \
                       f"2. Доступность сервиса эмбеддингов\n" \
                       f"3. Настройки сети и файрвола"
            elif "qdrant" in error_str.lower() or "vdb" in error_str.lower():
                return f"Не удалось подключиться к векторной базе данных.\n\nДетали: {detail if detail else error}\n\nПроверьте:\n" \
                       f"1. Правильность параметра vdb_url в запросе\n" \
                       f"2. Доступность Qdrant сервера\n" \
                       f"3. Настройки сети и файрвола"
            else:
                return f"Ошибка соединения: {detail if detail else error}"
        
        # Обработка HTTP ошибок
        if code == "http_error":
            if detail:
                return f"{error}\n\nДетали: {detail}"
            return error
        
        # Обработка других ошибок с деталями
        if detail and detail != error:
            # Если детали содержат полезную информацию, показываем их
            return f"{error}\n\n{detail}"
        
        return error
    
    def _remove_from_rag(self):
        """Удаление файлов из RAG."""
        irv_id = self.rag_manage_irv_id_var.get().strip()
        if not irv_id:
            messagebox.showwarning("Предупреждение", "Введите IRV ID")
            return
        
        # Очищаем поле результатов
        self.rag_manage_results.config(state=tk.NORMAL)
        self.rag_manage_results.delete(1.0, tk.END)
        self.rag_manage_results.config(state=tk.DISABLED)
        
        qdrant_config = self.config_manager.get("qdrant", {})
        request_data = {
            "vdb_url": qdrant_config.get("url", "http://localhost:6333"),
            "irv_id": irv_id,
            "action": "remove",
        }
        
        self.set_status("Удаление из RAG...", "blue")
        self.root.update()
        
        result = self.api_client.manage_rag_files(request_data)
        self._display_rag_manage_result(result)
    
    def _get_file_info(self):
        """Получение информации о файле в RAG."""
        irv_id = self.rag_manage_irv_id_var.get().strip()
        if not irv_id:
            messagebox.showwarning("Предупреждение", "Введите IRV ID")
            return
        
        # Очищаем поле результатов
        self.rag_manage_results.config(state=tk.NORMAL)
        self.rag_manage_results.delete(1.0, tk.END)
        self.rag_manage_results.config(state=tk.DISABLED)
        
        qdrant_config = self.config_manager.get("qdrant", {})
        request_data = {
            "vdb_url": qdrant_config.get("url", "http://localhost:6333"),
            "irv_id": irv_id,
            "action": "info",
        }
        
        self.set_status("Получение информации...", "blue")
        self.root.update()
        
        result = self.api_client.manage_rag_files(request_data)
        self._display_rag_manage_result(result)
    
    def _display_rag_manage_result(self, result: dict):
        """Отображение результата операции RAG."""
        self.rag_manage_results.config(state=tk.NORMAL)
        
        if result.get("success"):
            # Показываем весь ответ от сервера (как получает КФО)
            server_response = result.get("data", {})
            text = json.dumps(server_response, ensure_ascii=False, indent=2)
            self.rag_manage_results.insert(tk.END, text)
            self.set_status("Операция выполнена успешно", "green")
        else:
            # При ошибке показываем полный ответ от сервера
            server_response = result.get("data", result)
            text = json.dumps(server_response, ensure_ascii=False, indent=2)
            self.rag_manage_results.insert(tk.END, text)
            error_msg = self._format_error_message(result)
            self.set_status(f"Ошибка: {error_msg}", "red")
            messagebox.showerror("Ошибка", error_msg)
        
        self.rag_manage_results.config(state=tk.DISABLED)
    
    def _init_collections_tab(self):
        """Инициализация вкладки 'Коллекции'."""
        # Верхняя панель
        top_panel = ttk.Frame(self.collections_frame)
        top_panel.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(top_panel, text="Обновить список", command=self._refresh_collections).pack(side=tk.LEFT, padx=5)
        ttk.Button(top_panel, text="Проверить здоровье", command=self._check_qdrant_health).pack(side=tk.LEFT, padx=5)
        
        # Средняя панель - список коллекций
        middle_panel = ttk.Frame(self.collections_frame)
        middle_panel.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        collections_frame = ttk.LabelFrame(middle_panel, text="Коллекции", padding=10)
        collections_frame.pack(fill=tk.BOTH, expand=True)
        
        # Таблица коллекций
        columns = ("name", "points", "status", "vector_size")
        self.collections_tree = ttk.Treeview(collections_frame, columns=columns, show="headings", height=10)
        
        self.collections_tree.heading("name", text="Имя")
        self.collections_tree.heading("points", text="Точек")
        self.collections_tree.heading("status", text="Статус")
        self.collections_tree.heading("vector_size", text="Размер вектора")
        
        self.collections_tree.column("name", width=200)
        self.collections_tree.column("points", width=100)
        self.collections_tree.column("status", width=100)
        self.collections_tree.column("vector_size", width=150)
        
        scrollbar_collections = ttk.Scrollbar(collections_frame, orient=tk.VERTICAL, command=self.collections_tree.yview)
        self.collections_tree.config(yscrollcommand=scrollbar_collections.set)
        
        self.collections_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar_collections.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Нижняя панель - кнопки
        bottom_panel = ttk.Frame(self.collections_frame)
        bottom_panel.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(bottom_panel, text="Удалить коллекцию", command=self._delete_collection).pack(side=tk.LEFT, padx=5)
        
        # Результаты
        results_frame = ttk.LabelFrame(self.collections_frame, text="Результаты", padding=10)
        results_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.collections_results = scrolledtext.ScrolledText(results_frame, height=5, wrap=tk.WORD, state=tk.DISABLED)
        self.collections_results.pack(fill=tk.BOTH, expand=True)
    
    def _refresh_collections(self):
        """Обновление списка коллекций."""
        # Очищаем поле результатов
        self.collections_results.config(state=tk.NORMAL)
        self.collections_results.delete(1.0, tk.END)
        self.collections_results.config(state=tk.DISABLED)
        
        self.set_status("Получение списка коллекций...", "blue")
        self.root.update()
        
        try:
            qdrant_config = self.config_manager.get("qdrant", {})
            vdb_url = qdrant_config.get("url", "http://localhost:6333")
            logger.info(f"Запрос списка коллекций для vdb_url: {vdb_url}")
            
            result = self.api_client.get_collections(vdb_url)
            logger.debug(f"Результат запроса: {result}")
            
            if result.get("success"):
                data = result.get("data", {})
                content = data.get("content", {})
                
                # Проверяем структуру ответа
                if isinstance(content, dict):
                    collections = content.get("collections", [])
                    if not isinstance(collections, list):
                        # Возможно, collections - это сам список
                        if isinstance(content.get("collections"), list):
                            collections = content.get("collections")
                        else:
                            collections = []
                elif isinstance(content, list):
                    # Если content - это сам список коллекций
                    collections = content
                else:
                    collections = []
                
                logger.info(f"Найдено коллекций: {len(collections)}")
                
                # Очищаем дерево
                for item in self.collections_tree.get_children():
                    self.collections_tree.delete(item)
                
                # Добавляем коллекции
                for coll in collections:
                    if isinstance(coll, dict):
                        self.collections_tree.insert("", tk.END, values=(
                            coll.get("name", ""),
                            coll.get("points_count", 0),
                            coll.get("status", ""),
                            coll.get("vector_size", ""),
                        ))
                    else:
                        logger.warning(f"Неожиданный формат коллекции: {type(coll)}")
                
                self.set_status(f"Найдено коллекций: {len(collections)}", "green")
                
                # Выводим полный ответ от сервера (как получает КФО)
                self.collections_results.config(state=tk.NORMAL)
                server_response = result.get("data", {})
                text = json.dumps(server_response, ensure_ascii=False, indent=2)
                self.collections_results.insert(tk.END, text)
                self.collections_results.config(state=tk.DISABLED)
            else:
                # При ошибке показываем полный ответ от сервера
                error_msg = self._format_error_message(result)
                logger.error(f"Ошибка при получении коллекций: {error_msg}")
                self.set_status(f"Ошибка: {error_msg}", "red")
                self.collections_results.config(state=tk.NORMAL)
                server_response = result.get("data", result)
                text = json.dumps(server_response, ensure_ascii=False, indent=2)
                self.collections_results.insert(tk.END, text)
                self.collections_results.config(state=tk.DISABLED)
                messagebox.showerror("Ошибка", error_msg)
        except Exception as e:
            logger.exception(f"Исключение при обновлении коллекций: {e}")
            error_msg = f"Неожиданная ошибка: {str(e)}"
            self.set_status(f"Ошибка: {error_msg}", "red")
            self.collections_results.config(state=tk.NORMAL)
            self.collections_results.insert(tk.END, error_msg)
            self.collections_results.config(state=tk.DISABLED)
            messagebox.showerror("Ошибка", error_msg)
    
    def _check_qdrant_health(self):
        """Проверка здоровья Qdrant."""
        # Очищаем поле результатов
        self.collections_results.config(state=tk.NORMAL)
        self.collections_results.delete(1.0, tk.END)
        self.collections_results.config(state=tk.DISABLED)
        
        self.set_status("Проверка здоровья Qdrant...", "blue")
        self.root.update()
        
        qdrant_config = self.config_manager.get("qdrant", {})
        result = self.api_client.check_qdrant_health(qdrant_config.get("url", "http://localhost:6333"))
        
        self.collections_results.config(state=tk.NORMAL)
        
        if result.get("success"):
            # Показываем полный ответ от сервера (как получает КФО)
            server_response = result.get("data", {})
            text = json.dumps(server_response, ensure_ascii=False, indent=2)
            self.collections_results.insert(tk.END, text)
            if server_response.get("available"):
                self.set_status("Qdrant доступен", "green")
            else:
                self.set_status("Qdrant недоступен", "red")
        else:
            # При ошибке показываем полный ответ от сервера
            server_response = result.get("data", result)
            text = json.dumps(server_response, ensure_ascii=False, indent=2)
            self.collections_results.insert(tk.END, text)
            error_msg = self._format_error_message(result)
            self.set_status(f"Ошибка: {error_msg}", "red")
        
        self.collections_results.config(state=tk.DISABLED)
    
    def _delete_collection(self):
        """Удаление коллекции."""
        # Очищаем поле результатов
        self.collections_results.config(state=tk.NORMAL)
        self.collections_results.delete(1.0, tk.END)
        self.collections_results.config(state=tk.DISABLED)
        
        selection = self.collections_tree.selection()
        if not selection:
            messagebox.showwarning("Предупреждение", "Выберите коллекцию для удаления")
            return
        
        item = self.collections_tree.item(selection[0])
        collection_name = item["values"][0]
        
        if not messagebox.askyesno("Подтверждение", f"Удалить коллекцию '{collection_name}'?"):
            return
        
        self.set_status(f"Удаление коллекции {collection_name}...", "blue")
        self.root.update()
        
        qdrant_config = self.config_manager.get("qdrant", {})
        result = self.api_client.delete_collection(collection_name, qdrant_config.get("url", "http://localhost:6333"))
        
        self.collections_results.config(state=tk.NORMAL)
        if result.get("success"):
            # Показываем полный ответ от сервера (как получает КФО)
            server_response = result.get("data", {})
            text = json.dumps(server_response, ensure_ascii=False, indent=2)
            self.collections_results.insert(tk.END, text)
            self.set_status(f"Коллекция {collection_name} удалена", "green")
            self._refresh_collections()
        else:
            # При ошибке показываем полный ответ от сервера
            server_response = result.get("data", result)
            text = json.dumps(server_response, ensure_ascii=False, indent=2)
            self.collections_results.insert(tk.END, text)
            error_msg = self._format_error_message(result)
            self.set_status(f"Ошибка: {error_msg}", "red")
            messagebox.showerror("Ошибка", error_msg)
        self.collections_results.config(state=tk.DISABLED)
    
    def _init_settings_tab(self):
        """Инициализация вкладки 'Настройки'."""
        # Прокручиваемая область
        canvas = tk.Canvas(self.settings_frame)
        scrollbar = ttk.Scrollbar(self.settings_frame, orient=tk.VERTICAL, command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # API настройки
        api_frame = ttk.LabelFrame(scrollable_frame, text="API настройки", padding=10)
        api_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(api_frame, text="API URL:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.api_url_var = tk.StringVar(value=self.config_manager.get("api_url", "http://localhost:8000"))
        ttk.Entry(api_frame, textvariable=self.api_url_var, width=50).grid(row=0, column=1, pady=2, padx=5)
        
        ttk.Label(api_frame, text="CFX Emulator URL:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.cfx_emulator_url_var = tk.StringVar(value=self.config_manager.get("cfx_emulator_url", "http://localhost:8001"))
        ttk.Entry(api_frame, textvariable=self.cfx_emulator_url_var, width=50).grid(row=1, column=1, pady=2, padx=5)
        
        # LLM настройки
        llm_frame = ttk.LabelFrame(scrollable_frame, text="LLM настройки по умолчанию", padding=10)
        llm_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(llm_frame, text="URL:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.settings_llm_url_var = tk.StringVar(value=self.config_manager.get("llm", {}).get("url", ""))
        ttk.Entry(llm_frame, textvariable=self.settings_llm_url_var, width=50).grid(row=0, column=1, pady=2, padx=5)
        
        ttk.Label(llm_frame, text="API Key:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.settings_llm_api_key_var = tk.StringVar(value=self.config_manager.get("llm", {}).get("api_key", ""))
        ttk.Entry(llm_frame, textvariable=self.settings_llm_api_key_var, width=50, show="*").grid(row=1, column=1, pady=2, padx=5)
        
        ttk.Label(llm_frame, text="Модель:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.settings_llm_model_var = tk.StringVar(value=self.config_manager.get("llm", {}).get("model", "gpt-4o-mini"))
        ttk.Entry(llm_frame, textvariable=self.settings_llm_model_var, width=50).grid(row=2, column=1, pady=2, padx=5)
        
        ttk.Label(llm_frame, text="Температура:").grid(row=3, column=0, sticky=tk.W, pady=2)
        self.settings_temperature_var = tk.DoubleVar(value=self.config_manager.get("llm", {}).get("temperature", 0.2))
        ttk.Spinbox(llm_frame, from_=0.0, to=2.0, increment=0.1, textvariable=self.settings_temperature_var, width=47).grid(row=3, column=1, pady=2, padx=5)
        
        ttk.Label(llm_frame, text="Max Tokens:").grid(row=4, column=0, sticky=tk.W, pady=2)
        self.settings_max_tokens_var = tk.IntVar(value=self.config_manager.get("llm", {}).get("max_tokens", 8000))
        ttk.Spinbox(llm_frame, from_=1, to=32000, increment=1000, textvariable=self.settings_max_tokens_var, width=47).grid(row=4, column=1, pady=2, padx=5)
        
        ttk.Label(llm_frame, text="Системный промпт:").grid(row=5, column=0, sticky=tk.NW, pady=2)
        self.settings_system_prompt_text = scrolledtext.ScrolledText(llm_frame, height=6, width=50, wrap=tk.WORD)
        self.settings_system_prompt_text.insert(1.0, self.config_manager.get("llm", {}).get("system_prompt", ""))
        self.settings_system_prompt_text.grid(row=5, column=1, pady=2, padx=5, sticky=tk.W)
        
        # Embeddings настройки
        embed_frame = ttk.LabelFrame(scrollable_frame, text="Embeddings настройки по умолчанию", padding=10)
        embed_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(embed_frame, text="API Key:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.settings_embed_api_key_var = tk.StringVar(value=self.config_manager.get("embeddings", {}).get("api_key", ""))
        ttk.Entry(embed_frame, textvariable=self.settings_embed_api_key_var, width=50, show="*").grid(row=0, column=1, pady=2, padx=5)
        
        ttk.Label(embed_frame, text="URL:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.settings_embed_url_var = tk.StringVar(value=self.config_manager.get("embeddings", {}).get("url", ""))
        ttk.Entry(embed_frame, textvariable=self.settings_embed_url_var, width=50).grid(row=1, column=1, pady=2, padx=5)
        
        ttk.Label(embed_frame, text="Модель:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.settings_embed_model_var = tk.StringVar(value=self.config_manager.get("embeddings", {}).get("model", "Embeddings"))
        ttk.Entry(embed_frame, textvariable=self.settings_embed_model_var, width=50).grid(row=2, column=1, pady=2, padx=5)
        
        # Qdrant настройки
        qdrant_frame = ttk.LabelFrame(scrollable_frame, text="Qdrant настройки по умолчанию", padding=10)
        qdrant_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(qdrant_frame, text="URL:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.settings_qdrant_url_var = tk.StringVar(value=self.config_manager.get("qdrant", {}).get("url", "http://localhost:6333"))
        ttk.Entry(qdrant_frame, textvariable=self.settings_qdrant_url_var, width=50).grid(row=0, column=1, pady=2, padx=5)
        
        # Кнопки
        buttons_frame = ttk.Frame(scrollable_frame)
        buttons_frame.pack(fill=tk.X, padx=5, pady=10)
        
        ttk.Button(buttons_frame, text="Сохранить настройки", command=self._save_settings).pack(side=tk.LEFT, padx=5)
        ttk.Button(buttons_frame, text="Загрузить настройки", command=self._load_settings).pack(side=tk.LEFT, padx=5)
        
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def _save_settings(self):
        """Сохранение настроек."""
        self.config_manager.set("api_url", self.api_url_var.get())
        self.config_manager.set("cfx_emulator_url", self.cfx_emulator_url_var.get())
        self.config_manager.set("llm", {
            "url": self.settings_llm_url_var.get(),
            "api_key": self.settings_llm_api_key_var.get(),
            "model": self.settings_llm_model_var.get(),
            "temperature": self.settings_temperature_var.get(),
            "max_tokens": self.settings_max_tokens_var.get(),
            "system_prompt": self.settings_system_prompt_text.get(1.0, tk.END).strip(),
        })
        self.config_manager.set("embeddings", {
            "api_key": self.settings_embed_api_key_var.get(),
            "url": self.settings_embed_url_var.get(),
            "model": self.settings_embed_model_var.get(),
        })
        self.config_manager.set("qdrant", {
            "url": self.settings_qdrant_url_var.get(),
        })
        self.config_manager.save()
        
        # Обновляем URL клиента
        self.api_client.api_url = self.api_url_var.get()
        self.api_client.cfx_emulator_url = self.cfx_emulator_url_var.get()
        
        self.set_status("Настройки сохранены", "green")
        messagebox.showinfo("Успех", "Настройки сохранены")
    
    def _load_settings(self):
        """Загрузка настроек."""
        self.config_manager.load()
        self.api_url_var.set(self.config_manager.get("api_url", "http://localhost:8000"))
        self.cfx_emulator_url_var.set(self.config_manager.get("cfx_emulator_url", "http://localhost:8001"))
        
        llm_config = self.config_manager.get("llm", {})
        self.settings_llm_url_var.set(llm_config.get("url", ""))
        self.settings_llm_api_key_var.set(llm_config.get("api_key", ""))
        self.settings_llm_model_var.set(llm_config.get("model", "gpt-4o-mini"))
        self.settings_temperature_var.set(llm_config.get("temperature", 0.2))
        self.settings_max_tokens_var.set(llm_config.get("max_tokens", 8000))
        # Загружаем системный промпт
        system_prompt = llm_config.get("system_prompt", "")
        self.settings_system_prompt_text.delete(1.0, tk.END)
        self.settings_system_prompt_text.insert(1.0, system_prompt)
        
        embed_config = self.config_manager.get("embeddings", {})
        self.settings_embed_api_key_var.set(embed_config.get("api_key", ""))
        self.settings_embed_url_var.set(embed_config.get("url", ""))
        self.settings_embed_model_var.set(embed_config.get("model", "Embeddings"))
        
        qdrant_config = self.config_manager.get("qdrant", {})
        self.settings_qdrant_url_var.set(qdrant_config.get("url", "http://localhost:6333"))
        
        self.set_status("Настройки загружены", "green")
    
    def _init_user_tab(self):
        """Инициализация вкладки 'Пользователь'."""
        from desktop.cfx_emulator import storage
        
        # Основной фрейм
        main_frame = ttk.Frame(self.user_frame)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Фрейм с полями
        fields_frame = ttk.LabelFrame(main_frame, text="Информация о пользователе", padding=10)
        fields_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # ID пользователя
        ttk.Label(fields_frame, text="ID:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.user_id_var = tk.StringVar()
        ttk.Entry(fields_frame, textvariable=self.user_id_var, width=50).grid(row=0, column=1, pady=5, padx=5, sticky=tk.W+tk.E)
        
        # Имя пользователя
        ttk.Label(fields_frame, text="Имя:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.user_name_var = tk.StringVar()
        ttk.Entry(fields_frame, textvariable=self.user_name_var, width=50).grid(row=1, column=1, pady=5, padx=5, sticky=tk.W+tk.E)
        
        # Email
        ttk.Label(fields_frame, text="Email:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.user_email_var = tk.StringVar()
        ttk.Entry(fields_frame, textvariable=self.user_email_var, width=50).grid(row=2, column=1, pady=5, padx=5, sticky=tk.W+tk.E)
        
        # Должность
        ttk.Label(fields_frame, text="Должность:").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.user_post_var = tk.StringVar()
        ttk.Entry(fields_frame, textvariable=self.user_post_var, width=50).grid(row=3, column=1, pady=5, padx=5, sticky=tk.W+tk.E)
        
        fields_frame.columnconfigure(1, weight=1)
        
        # Кнопки
        buttons_frame = ttk.Frame(main_frame)
        buttons_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(buttons_frame, text="Загрузить", command=self._load_user_info).pack(side=tk.LEFT, padx=5)
        ttk.Button(buttons_frame, text="Сохранить", command=self._save_user_info).pack(side=tk.LEFT, padx=5)
        
        # Загружаем информацию о пользователе при инициализации
        self._load_user_info()
    
    def _load_user_info(self):
        """Загрузка информации о пользователе из хранилища эмулятора."""
        from desktop.cfx_emulator import storage
        
        try:
            user_info = storage.get_user_info()
            self.user_id_var.set(user_info.get("id", ""))
            self.user_name_var.set(user_info.get("name", ""))
            self.user_email_var.set(user_info.get("email", ""))
            self.user_post_var.set(user_info.get("userPost", ""))
            self.set_status("Информация о пользователе загружена", "green")
        except Exception as e:
            logger.exception(f"Ошибка загрузки информации о пользователе: {e}")
            self.set_status(f"Ошибка загрузки: {e}", "red")
            messagebox.showerror("Ошибка", f"Не удалось загрузить информацию о пользователе: {e}")
    
    def _save_user_info(self):
        """Сохранение информации о пользователе в хранилище эмулятора."""
        from desktop.cfx_emulator import storage
        
        try:
            user_info = {
                "id": self.user_id_var.get().strip(),
                "name": self.user_name_var.get().strip(),
                "email": self.user_email_var.get().strip(),
                "userPost": self.user_post_var.get().strip(),
            }
            
            # Проверяем обязательные поля
            if not user_info["id"]:
                messagebox.showwarning("Предупреждение", "Поле 'ID' обязательно для заполнения")
                return
            
            storage.save_user_info(user_info)
            self.set_status("Информация о пользователе сохранена", "green")
            messagebox.showinfo("Успех", "Информация о пользователе сохранена")
        except Exception as e:
            logger.exception(f"Ошибка сохранения информации о пользователе: {e}")
            self.set_status(f"Ошибка сохранения: {e}", "red")
            messagebox.showerror("Ошибка", f"Не удалось сохранить информацию о пользователе: {e}")

