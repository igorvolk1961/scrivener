"""
Вспомогательные UI компоненты.
"""

import tkinter as tk
from tkinter import ttk, scrolledtext


class ChatHistoryWidget(scrolledtext.ScrolledText):
    """Виджет для отображения истории чата с прокруткой."""
    
    def __init__(self, parent, **kwargs):
        super().__init__(parent, wrap=tk.WORD, state=tk.DISABLED, **kwargs)
        self.tag_configure("user", foreground="blue")
        self.tag_configure("assistant", foreground="green")
        self.tag_configure("error", foreground="red")
    
    def add_message(self, role: str, content: str):
        """Добавляет сообщение в историю."""
        self.config(state=tk.NORMAL)
        tag = role if role in ["user", "assistant", "error"] else "user"
        self.insert(tk.END, f"{role.upper()}: {content}\n\n", tag)
        self.config(state=tk.DISABLED)
        self.see(tk.END)
    
    def clear(self):
        """Очищает историю."""
        self.config(state=tk.NORMAL)
        self.delete(1.0, tk.END)
        self.config(state=tk.DISABLED)


class StatusBar(ttk.Frame):
    """Строка статуса внизу окна."""
    
    def __init__(self, parent):
        super().__init__(parent)
        self.status_label = ttk.Label(self, text="Готов", relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
    
    def set_status(self, message: str, color: str = "black"):
        """Устанавливает статус."""
        self.status_label.config(text=message, foreground=color)

