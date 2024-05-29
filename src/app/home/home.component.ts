import { ChangeDetectorRef, Component, ElementRef, OnInit, ViewChild, ViewEncapsulation } from '@angular/core';
import { Router } from '@angular/router';
import { PrimeNGConfig } from 'primeng/api';

@Component({
  selector: 'app-home',
  templateUrl: './home.component.html',
  styleUrls: ['./home.component.css'],
  encapsulation: ViewEncapsulation.None
})

export class HomeComponent implements OnInit {
  @ViewChild('sidebar', {static: true}) sidebar: any;
  public toggleShow : boolean = true;
  public view : string = '0';
  public optionNav = 0;
  public spinner : string = '0';
  public username : any;
  public bLogger = false;
  public avatar : any = "";

  public constructor(
    private primengConfig: PrimeNGConfig, private router: Router, private cdRef: ChangeDetectorRef) {
  }

  ngOnInit() {
    this.primengConfig.ripple = true;
  
    var rol = localStorage.getItem('role');
    var email = localStorage.getItem('email');
    this.avatar = localStorage.getItem('avatar');
    
    if (!this.isNullOrUndefined(rol) && !this.isNullOrUndefined(email)) {
        this.bLogger = true;
        this.username = localStorage.getItem('username');
    }else{
      //comentado para que se pueda probar desde Netlify mientras se sube el backend a un server
      //this.router.navigate(['/login'], { replaceUrl: true });
    }

    localStorage.setItem('view', '0');
    localStorage.setItem('spinner', '0');
    
    this.hideToggle();
  }

  public collapseHide(){
    $('#navbarNav').removeClass('show');
  }

  public isNullOrUndefined(item : any){
    if(item == null || item == "" || item == 0 || item.length == 0 || item.length == undefined){
      return true;
    }

    return false;
  }

  public Gotologin(){
    this.router.navigate(['/login'], { replaceUrl: true });
  }

  public Gotoregister(){
    this.router.navigate(['/register'], { replaceUrl: true });
  }

  public Gotohome(){
    this.router.navigate(['/home/post'], { replaceUrl: true });
  }

  public Gotorequest(){
    this.router.navigate(['/request'], { replaceUrl: true });
  }

  public showToggle(){
    if(this.sidebar.nativeElement.classList.contains('hide-block')){
      this.toggleShow = true;
      this.sidebar.nativeElement.classList.remove('hide-block');
    }else{
      this.toggleShow = false;
      this.sidebar.nativeElement.classList.add('hide-block');
    }
  }

  public hideToggle(){
    if(window.innerWidth <= 768){
      this.toggleShow = false;
      this.sidebar.nativeElement.classList.add('hide-block');
    }
  }

  public activeNav(numero : number){
    this.optionNav = numero;
  }

  public logout(){
		localStorage.removeItem('token');
  	localStorage.removeItem('email');
  	localStorage.removeItem('role');
  	localStorage.removeItem('username');
  	localStorage.removeItem('avatar');
  	localStorage.setItem('isLoggedIn', "false");  

    this.router.navigateByUrl('/login');
	}

  ngAfterViewChecked()
  {
    this.view = localStorage.getItem('view')!;
    this.spinner = localStorage.getItem('spinner')!;
    this.cdRef.detectChanges();
  }

}


