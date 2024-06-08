import { Component } from '@angular/core';

@Component({
  selector: 'app-contact-us',
  templateUrl: './contact-us.component.html',
  styleUrls: ['./contact-us.component.css']
})
export class ContactUsComponent {
  public bLogger = false;

  constructor() { }

  ngOnInit() {

      var rol = localStorage.getItem('role');
      var email = localStorage.getItem('email');
      
      if (!this.isNullOrUndefined(rol) && !this.isNullOrUndefined(email)) {
          this.bLogger = true;
      }
  }

  public isNullOrUndefined(item : any){
    if(item == null || item == "" || item == 0 || item.length == 0 || item.length == undefined){
      return true;
    }

    return false;
  }
}
