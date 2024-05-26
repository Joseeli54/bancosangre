import { Component, OnInit } from '@angular/core';
import { UserServices } from '../services/userservices.services';
import { ActivatedRoute, ActivatedRouteSnapshot, RouterStateSnapshot, Router } from '@angular/router';
import { FormControl, FormGroup, FormBuilder, Validators } from '@angular/forms';
import { SweetAlertOptions } from 'sweetalert2';
import Swal from 'sweetalert2';
import $ from 'jquery';

@Component({
  selector: 'app-login',
  templateUrl: './login.component.html',
  styleUrls: ['./login.component.css']
})
export class LoginComponent {
  form!: FormGroup;
  loading = false;
  submitted = false;
  before =false;
  error = ''
  name = 'Usuario'
  public input = {
    email: '',
    password: '',
    recuerdame: false,
    codigo2FA: '',
    authCSRF_2FA: '',
    email_recuperar: ''
  }
  public cargando = false
  idUser: string = '';
  password: string = '';
  private localStorageService;

  constructor(
      private formBuilder: FormBuilder,
      private route: ActivatedRoute,
      private router: Router,
      public userService: UserServices
  ) { 
    this.localStorageService = localStorage;
  }

  ngOnInit() {
    var rol = localStorage.getItem('role');
    var email = localStorage.getItem('email');
    if (!this.isNullOrUndefined(rol) && !this.isNullOrUndefined(email)) {
      this.router.navigate(['/home/option-list'], { replaceUrl: true });
    }
  }

  public isNullOrUndefined(element : any){
    if(element === undefined || element === null || element === ''){
      return true;
    }

    return false;
  }

  login() {
    let data;

    if(this.idUser.indexOf('@') !== -1){
      data = {
        "email": this.idUser,
        "password": this.password,
      };
    }else{
      data = {
        "username": this.idUser,
        "password": this.password,
      };
    }

    $('#login_button').html("<li class='fa fa-spinner fa-spin fa-1x'> </li>");

    this.userService
      .postUrl('login', data)
      .then(response => {
              console.log("Login successful");  
              
              localStorage.setItem('isLoggedIn', "true"); 
              localStorage.setItem('token', response.token);
              localStorage.setItem('email', response.user.email);
              localStorage.setItem('username', response.user.username);
              localStorage.setItem('role', response.user.role);
              localStorage.setItem('avatar', response.user.avatar);

              $('#login_button').html("Login");

              this.messageSuccessfully();

            })
      .catch(data =>{
              $('#login_button').html("Login");
              this.errorOcurred(data.error.err.message)
      });
    
  }

  private messageSuccessfully() {
    let config: SweetAlertOptions = {
      title: 'Your session is starting...',
      icon: 'success',
      showConfirmButton: false,
      timer: 2500
    };

    Swal.fire(config).then(result => {
      this.router.navigate(['/home/option-list'], { replaceUrl: true });
    });
  }

  private errorOcurred(message : any) {
    let config: SweetAlertOptions = {
      title: message,
      icon: 'error',
      showConfirmButton: true
    };

    Swal.fire(config).then(result => {
    });
  }

  public backClicked() {
    window.history.back();
  }

  public abrirModalRecuperarUsuario(){

  }

  public cancelarRecuerdame(){

  }

}
